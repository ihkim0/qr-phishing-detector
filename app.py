"""QR Phishing Detector - Flask Web Application

이 애플리케이션은 사용자가 QR 코드 이미지를 업로드하거나 URL을 직접 입력하여
QR 피싱(큐싱, Qshing) 여부 및 위험도를 실시간으로 분석하고,
분석 결과를 직관적인 웹 대시보드로 렌더링합니다.
"""

import os
import uuid
import logging
from typing import Dict, Any, List

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_from_directory
)
from werkzeug.utils import secure_filename

# 프로젝트 내부 모듈 연동
from src.detector import analyze_qr_image, analyze_url, DEFAULT_MODEL_PATH
from src.database import get_recent_scans, init_db, clear_history

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Flask 애플리케이션 인스턴스 초기화
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# 업로드 파일 디렉토리 및 보안 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 최대 16MB 파일 허용

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}


def allowed_file(filename: str) -> bool:
    """업로드된 파일의 확장자가 허용된 이미지 확장자인지 검증합니다.

    Args:
        filename: 원본 파일명

    Returns:
        bool: 허용 여부
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.before_request
def setup_database_once():
    """애플리케이션 요청 전 DB 테이블 초기화 보장"""
    init_db()


@app.route("/", methods=["GET", "POST"])
def index():
    """메인 대시보드 페이지 라우트

    GET: 최근 검사 이력과 함께 기본 업로드 대시보드 렌더링
    POST: 업로드된 QR 이미지 또는 직접 입력된 URL을 받아 피싱 분석 수행 후 결과 렌더링
    """
    analysis_result = None
    uploaded_image_url = None

    if request.method == "POST":
        analysis_mode = request.form.get("analysis_mode", "qr")

        # 1. QR 이미지 파일 분석 모드
        if analysis_mode == "qr":
            if "qr_file" not in request.files:
                flash("QR 이미지 파일을 선택해 주세요.", "warning")
                return redirect(request.url)

            file = request.files["qr_file"]
            if file.filename == "":
                flash("선택된 파일이 없습니다.", "warning")
                return redirect(request.url)

            if file and allowed_file(file.filename):
                # 안전한 고유 파일명 생성 (충돌 방지)
                original_name = secure_filename(file.filename) or "qrcode.png"
                ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else "png"
                unique_filename = f"{uuid.uuid4().hex[:12]}_{original_name}"
                saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

                try:
                    file.save(saved_path)
                    logger.info(f"업로드 파일 저장 완료: {saved_path}")

                    # 통합 탐지 엔진 호출
                    analysis_result = analyze_qr_image(
                        image_path=saved_path,
                        model_path=DEFAULT_MODEL_PATH,
                        save_to_db=True
                    )
                    uploaded_image_url = url_for("uploaded_file", filename=unique_filename)

                except Exception as e:
                    logger.error(f"이미지 분석 처리 중 오류 발생: {e}")
                    flash(f"이미지 분석 처리 중 오류가 발생했습니다: {e}", "danger")
            else:
                flash("지원되지 않는 파일 형식입니다. (PNG, JPG, JPEG, WEBP 등 이미지 파일만 허용)", "danger")

        # 2. URL 직접 입력 분석 모드 (테스트 및 편의용)
        elif analysis_mode == "url":
            target_url = request.form.get("direct_url", "").strip()
            if not target_url:
                flash("분석할 URL을 입력해 주세요.", "warning")
            else:
                try:
                    analysis_result = analyze_url(
                        url=target_url,
                        model_path=DEFAULT_MODEL_PATH,
                        save_to_db=True
                    )
                except Exception as e:
                    logger.error(f"URL 직접 분석 중 오류 발생: {e}")
                    flash(f"URL 분석 처리 중 오류가 발생했습니다: {e}", "danger")

    # 최근 검사 이력 10건 조회
    try:
        recent_scans = get_recent_scans(limit=10)
    except Exception as e:
        logger.error(f"최근 이력 조회 실패: {e}")
        recent_scans = []

    return render_template(
        "index.html",
        result=analysis_result,
        uploaded_image_url=uploaded_image_url,
        recent_scans=recent_scans
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    """업로드된 이미지 파일을 브라우저 미리보기에 서빙합니다."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ============================================================================
# REST API 엔드포인트 (외부 연동 및 테스트용)
# ============================================================================
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """QR 이미지 업로드 또는 JSON URL 전달 시 분석 결과를 반환하는 REST API"""
    # 1. 파일 업로드 요청 처리
    if "qr_file" in request.files:
        file = request.files["qr_file"]
        if file and allowed_file(file.filename):
            unique_filename = f"api_{uuid.uuid4().hex[:12]}.png"
            saved_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(saved_path)
            res = analyze_qr_image(saved_path, model_path=DEFAULT_MODEL_PATH, save_to_db=True)
            return jsonify(res)
        return jsonify({"success": False, "error": "유효하지 않은 이미지 파일입니다."}), 400

    # 2. JSON URL 분석 요청 처리
    data = request.get_json(silent=True) or {}
    url = data.get("url")
    if url:
        res = analyze_url(url, model_path=DEFAULT_MODEL_PATH, save_to_db=True)
        return jsonify(res)

    return jsonify({"success": False, "error": "'qr_file' 파일 또는 'url' JSON 필드가 필요합니다."}), 400


@app.route("/api/history", methods=["GET"])
def api_history():
    """최근 검사 이력을 반환하는 API"""
    limit = request.args.get("limit", default=10, type=int)
    history = get_recent_scans(limit=limit)
    return jsonify({"success": True, "count": len(history), "history": history})


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    """전체 검사 이력을 초기화하는 API"""
    success = clear_history()
    if request.is_json:
        return jsonify({"success": success})
    flash("검사 이력이 모두 초기화되었습니다.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  QR Phishing Detector Flask 웹 서버 구동 중...  ")
    logger.info("  접속 주소: http://127.0.0.1:5000 또는 http://localhost:5000  ")
    logger.info("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
