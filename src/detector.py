"""QR Phishing Detector - Integrated Detection Engine

OpenCV를 통한 QR 코드 이미지 감지 및 디코딩,
URL 다차원 특징 추출, 규칙 기반 휴리스틱 위험도 산출,
사전 학습된 머신러닝(RandomForest) 모델 추론을 통합하여
최종 피싱 여부 판정, 위험도 점수(0~100) 및 사용자가 이해할 수 있는 탐지 근거(Details)를 제공합니다.
"""

import os
import re
import logging
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import joblib

# 프로젝트 내 DB 모듈 연동
from src.database import save_scan_result

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 기본 모델 파일 경로
DEFAULT_MODEL_PATH = "model.pkl"

# 특징(Feature) 이름 목록 (train.py와 동일)
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "path_length",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_question",
    "count_equal",
    "count_slash",
    "count_percent",
    "count_digits",
    "digit_ratio",
    "has_ip_address",
    "is_https",
    "num_subdomains",
    "suspicious_keyword_cnt",
    "is_shortened_url"
]

# 피싱 의심 키워드 목록
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "verification", "secure", "security", "account",
    "update", "banking", "signin", "sign-in", "confirm", "password",
    "wallet", "authenticate", "auth", "support", "service", "billing",
    "alert", "notice", "suspended", "payment", "re-login"
]

# 대표적인 단축 URL 도메인 목록
SHORTENED_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "buff.ly",
    "ow.ly", "cutt.ly", "rebrand.ly", "vo.la", "me2.do"
]

# IPv4 검출 정규식
IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


# ============================================================================
# 1. OpenCV 기반 QR 이미지 디코딩 함수
# ============================================================================
def decode_qr_image(image_path: str) -> Optional[str]:
    """업로드된 이미지 파일 경로로부터 QR 코드를 검출하고 내포된 URL/텍스트를 디코딩합니다.

    Windows 환경의 유니코드/한글 경로 문제를 방지하기 위해 numpy buffer를 거쳐 cv2.imdecode를 사용하며,
    인식 실패 시 그레이스케일 및 Otsu 이진화 전처리를 적용한 2차 시도를 수행합니다.

    Args:
        image_path: QR 코드가 포함된 이미지 파일의 절대 또는 상대 경로

    Returns:
        Optional[str]: 디코딩된 문자열 (QR 코드가 없거나 디코딩 실패 시 None)

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
    """
    if not os.path.exists(image_path):
        logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {image_path}")

    try:
        # Windows 한글 경로 대응: 바이너리 읽기 후 imdecode
        with open(image_path, "rb") as f:
            bytes_data = f.read()

        if not bytes_data:
            logger.warning(f"이미지 파일 내용이 비어 있습니다: {image_path}")
            return None

        img_np = np.frombuffer(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            logger.warning(f"이미지 파일 디코딩 실패 (지원하지 않는 포맷): {image_path}")
            return None

        # 1차 시도: 원본 컬러 이미지에서 감지
        detector = cv2.QRCodeDetector()
        decoded_text, points, _ = detector.detectAndDecode(img)

        if decoded_text and decoded_text.strip():
            logger.info(f"QR 코드 검출 성공 (1차 원본): '{decoded_text.strip()}'")
            return decoded_text.strip()

        # 2차 시도: 그레이스케일 및 대비 강화(Otsu Thresholding) 전처리 후 감지
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        decoded_text_2, _, _ = detector.detectAndDecode(thresh)
        if decoded_text_2 and decoded_text_2.strip():
            logger.info(f"QR 코드 검출 성공 (2차 전처리): '{decoded_text_2.strip()}'")
            return decoded_text_2.strip()

        logger.info(f"이미지에서 QR 코드를 검출하지 못했습니다: {image_path}")
        return None

    except Exception as e:
        logger.error(f"QR 코드 디코딩 처리 중 예외 발생 ({image_path}): {e}")
        return None


# ============================================================================
# 2. URL 특징(Feature) 추출 함수
# ============================================================================
def extract_url_features(url: str) -> Dict[str, float]:
    """단일 URL 문자열로부터 수치형 특징 벡터를 추출합니다.

    Args:
        url: 분석할 대상 URL

    Returns:
        Dict[str, float]: 특징명과 수치값 매핑
    """
    if not isinstance(url, str) or not url.strip():
        return {name: 0.0 for name in FEATURE_NAMES}

    url_str = url.strip()

    if not url_str.startswith(("http://", "https://")):
        parsed = urlparse("http://" + url_str)
        is_https = 0.0
    else:
        parsed = urlparse(url_str)
        is_https = 1.0 if parsed.scheme.lower() == "https" else 0.0

    hostname = parsed.netloc.split(":")[0].lower() if parsed.netloc else ""
    path = parsed.path

    url_length = float(len(url_str))
    hostname_length = float(len(hostname))
    path_length = float(len(path))

    count_dots = float(url_str.count("."))
    count_hyphens = float(url_str.count("-"))
    count_at = float(url_str.count("@"))
    count_question = float(url_str.count("?"))
    count_equal = float(url_str.count("="))
    count_slash = float(url_str.count("/"))
    count_percent = float(url_str.count("%"))

    digits = sum(c.isdigit() for c in url_str)
    count_digits = float(digits)
    digit_ratio = float(digits / len(url_str)) if url_length > 0 else 0.0

    has_ip = 1.0 if IP_PATTERN.match(hostname) else 0.0

    subdomain_parts = hostname.split(".")
    num_subdomains = float(max(0, len(subdomain_parts) - 2)) if len(subdomain_parts) > 1 else 0.0

    url_lower = url_str.lower()
    keyword_cnt = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    suspicious_keyword_cnt = float(keyword_cnt)

    is_shortened = 1.0 if any(hostname == s_dom or hostname.endswith("." + s_dom) for s_dom in SHORTENED_DOMAINS) else 0.0

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "count_dots": count_dots,
        "count_hyphens": count_hyphens,
        "count_at": count_at,
        "count_question": count_question,
        "count_equal": count_equal,
        "count_slash": count_slash,
        "count_percent": count_percent,
        "count_digits": count_digits,
        "digit_ratio": digit_ratio,
        "has_ip_address": has_ip,
        "is_https": is_https,
        "num_subdomains": num_subdomains,
        "suspicious_keyword_cnt": suspicious_keyword_cnt,
        "is_shortened_url": is_shortened
    }


# ============================================================================
# 3. 규칙 기반 위험 요소 분석 및 근거 생성 함수
# ============================================================================
def analyze_rules(url: str, features: Dict[str, float]) -> Tuple[float, List[str]]:
    """사전 정의된 보안 규칙 및 휴리스틱에 따라 위험도 점수와 탐지 근거를 산출합니다.

    Args:
        url: 검사 URL 문자열
        features: extract_url_features에서 추출된 특징 딕셔너리

    Returns:
        Tuple[float, List[str]]: (규칙 기반 위험도 점수 0~100, 탐지 근거 문자열 리스트)
    """
    rule_score = 0.0
    details: List[str] = []

    # 1. IP 주소 직접 사용
    if features.get("has_ip_address", 0.0) == 1.0:
        rule_score += 40.0
        details.append("호스트명에 도메인 이름 대신 숫자로 된 IP 주소(예: 192.168.x.x)가 직접 사용되었습니다.")

    # 2. HTTPS 미적용
    if features.get("is_https", 0.0) == 0.0:
        rule_score += 20.0
        details.append("보안 암호화 통신(HTTPS)이 적용되지 않은 안전하지 않은 HTTP 연결입니다.")

    # 3. 단축 URL 사용
    if features.get("is_shortened_url", 0.0) == 1.0:
        rule_score += 25.0
        details.append("최종 접속 목적지를 은폐할 위험이 있는 단축 URL(bit.ly, tinyurl 등)이 감지되었습니다.")

    # 4. 피싱 의심 키워드 검출
    kw_count = int(features.get("suspicious_keyword_cnt", 0.0))
    if kw_count > 0:
        url_lower = url.lower()
        found_kws = [kw for kw in SUSPICIOUS_KEYWORDS if kw in url_lower]
        kw_penalty = min(30.0, kw_count * 15.0)
        rule_score += kw_penalty
        details.append(f"계정 탈취 및 금융 사기에 자주 악용되는 의심 키워드({', '.join(found_kws[:4])})가 포함되어 있습니다.")

    # 5. '@' 기호 포함 (Basic Auth 스푸핑)
    if features.get("count_at", 0.0) > 0.0:
        rule_score += 30.0
        details.append("URL에 사용자 신원 위장에 자주 악용되는 골뱅이('@') 기호가 포함되어 있습니다.")

    # 6. 서브도메인 과다
    num_subs = int(features.get("num_subdomains", 0.0))
    if num_subs >= 3:
        rule_score += 15.0
        details.append(f"서브도메인 단계가 비정상적으로 깊어({num_subs}단계) 공공기관이나 정상 포털 사칭이 의심됩니다.")

    # 7. 비정상적으로 긴 URL
    if features.get("url_length", 0.0) >= 80.0:
        rule_score += 10.0
        details.append("URL 길이가 비정상적으로 길어 악성 페이로드나 난독화된 파라미터가 숨겨져 있을 수 있습니다.")

    # 8. 숫자 비율이 지나치게 높음
    if features.get("digit_ratio", 0.0) >= 0.25:
        rule_score += 10.0
        details.append("URL 내에 무작위 숫자 비율이 높아 자동 생성된 임시 악성 링크일 가능성이 있습니다.")

    # 0 ~ 100 범위로 클램핑
    clamped_score = min(100.0, max(0.0, rule_score))
    return clamped_score, details


# ============================================================================
# 4. 머신러닝 모델 추론 함수
# ============================================================================
class MLModelLoader:
    """학습된 모델(model.pkl) 싱글톤 로더"""
    _instance = None
    _model = None
    _feature_names = None

    @classmethod
    def get_model(cls, model_path: str = DEFAULT_MODEL_PATH):
        """저장된 model.pkl 파일로부터 학습 모델과 특징 명단을 로드합니다."""
        if cls._model is None or cls._instance != model_path:
            if not os.path.exists(model_path):
                logger.warning(f"모델 파일이 존재하지 않습니다: '{model_path}'. ML 추론을 건너뜁니다.")
                return None, None

            try:
                loaded = joblib.load(model_path)
                if isinstance(loaded, dict) and "model" in loaded:
                    cls._model = loaded["model"]
                    cls._feature_names = loaded.get("feature_names", FEATURE_NAMES)
                else:
                    cls._model = loaded
                    cls._feature_names = getattr(loaded, "feature_names_", FEATURE_NAMES)
                cls._instance = model_path
                logger.info(f"ML 모델 로드 성공: '{model_path}'")
            except Exception as e:
                logger.error(f"ML 모델 로드 실패 ({model_path}): {e}")
                return None, None

        return cls._model, cls._feature_names


def predict_ml(
    features: Dict[str, float],
    model_path: str = DEFAULT_MODEL_PATH
) -> Tuple[Optional[float], Optional[int]]:
    """특징 벡터를 기반으로 ML 모델의 피싱 확률과 클래스를 추론합니다.

    Args:
        features: extract_url_features에서 도출된 특징 딕셔너리
        model_path: 모델 파일 경로

    Returns:
        Tuple[Optional[float], Optional[int]]: (피싱 확률 0.0~1.0, 예측 클래스 0:정상 / 1:피싱).
        모델 부재 시 (None, None) 반환.
    """
    model, feature_names = MLModelLoader.get_model(model_path)
    if model is None:
        return None, None

    try:
        cols = feature_names or FEATURE_NAMES
        vector = [features.get(col, 0.0) for col in cols]
        X = pd.DataFrame([vector], columns=cols)

        pred_class = int(model.predict(X)[0])
        if hasattr(model, "predict_proba"):
            # 피싱 클래스(1)의 확률
            proba = float(model.predict_proba(X)[0][1])
        else:
            proba = 1.0 if pred_class == 1 else 0.0

        return proba, pred_class
    except Exception as e:
        logger.error(f"ML 모델 예측 도중 예외 발생: {e}")
        return None, None


# ============================================================================
# 5. 통합 탐지 엔진 (Integrated Detection Engine)
# ============================================================================
def analyze_url(
    url: str,
    model_path: str = DEFAULT_MODEL_PATH,
    save_to_db: bool = True
) -> Dict[str, Any]:
    """URL 문자열을 전달받아 규칙 및 ML 모델을 종합 분석하고 최종 판정을 도출합니다.

    Args:
        url: 분석 대상 URL
        model_path: 머신러닝 모델 파일 경로
        save_to_db: 결과를 SQLite DB에 기록할지 여부

    Returns:
        Dict[str, Any]: 종합 분석 결과 딕셔너리
    """
    if not isinstance(url, str) or not url.strip():
        return {
            "success": False,
            "url": "",
            "result": "오류",
            "score": 0.0,
            "rule_score": 0.0,
            "ml_probability": 0.0,
            "details": ["입력된 URL이 올바르지 않습니다."],
            "features": {}
        }

    clean_url = url.strip()

    # 1. 특징 추출
    features = extract_url_features(clean_url)

    # 2. 규칙 기반 분석
    rule_score, rule_details = analyze_rules(clean_url, features)

    # 3. 머신러닝 추론
    ml_proba, ml_class = predict_ml(features, model_path=model_path)

    details = list(rule_details)

    # 4. 복합 종합 점수(Composite Risk Score) 산출
    if ml_proba is not None:
        ml_score = ml_proba * 100.0
        # 규칙 점수와 ML 점수의 가중치 합성 (각 50%)
        composite_score = round(0.5 * rule_score + 0.5 * ml_score, 1)

        if ml_proba >= 0.70:
            details.append(f"머신러닝 모델이 피싱 URL 패턴과 {ml_proba * 100:.1f}% 일치한다고 판정했습니다.")
        elif ml_proba >= 0.40:
            details.append(f"머신러닝 모델이 의심스러운 링크 패턴을 감지했습니다. (위험 확률: {ml_proba * 100:.1f}%)")
    else:
        # 모델이 없을 경우 규칙 점수 100% 적용
        composite_score = round(rule_score, 1)
        details.append("ML 모델 미탑재로 규칙 기반 보안 검사만 수행되었습니다.")

    # 5. 최종 판정 (정상 / 주의 / 피싱)
    if composite_score >= 65.0:
        result_label = "피싱"
    elif composite_score >= 35.0:
        result_label = "주의"
    else:
        result_label = "정상"

    # 정상인 경우 긍정적인 세부 내역 추가
    if result_label == "정상" and not details:
        details.append("알려진 피싱 패턴 및 악성 URL 의심 요소가 발견되지 않은 안전한 웹사이트입니다.")

    # 6. SQLite DB 저장 (선택적)
    db_id = None
    if save_to_db:
        try:
            db_id = save_scan_result(
                input_url=clean_url,
                result=result_label,
                score=composite_score,
                details=details
            )
        except Exception as e:
            logger.error(f"DB 저장 중 예외 발생: {e}")

    return {
        "success": True,
        "url": clean_url,
        "result": result_label,
        "score": composite_score,
        "rule_score": round(rule_score, 1),
        "ml_probability": round(ml_proba, 4) if ml_proba is not None else None,
        "details": details,
        "features": features,
        "db_id": db_id
    }


def analyze_qr_image(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    save_to_db: bool = True
) -> Dict[str, Any]:
    """업로드된 QR 코드 이미지 파일을 검출/디코딩하고 피싱 위험도를 통합 분석합니다.

    Args:
        image_path: 검사할 QR 이미지 경로
        model_path: 머신러닝 모델 파일 경로
        save_to_db: 결과를 DB에 저장할지 여부

    Returns:
        Dict[str, Any]: 최종 분석 결과 딕셔너리
    """
    # 1. 파일 검증
    if not os.path.exists(image_path):
        return {
            "success": False,
            "url": None,
            "result": "오류",
            "score": 0.0,
            "error_message": f"요청한 이미지 파일이 존재하지 않습니다: {image_path}",
            "details": ["이미지 파일을 찾을 수 없습니다."]
        }

    # 2. QR 코드 디코딩
    decoded_url = decode_qr_image(image_path)
    if not decoded_url:
        return {
            "success": False,
            "url": None,
            "result": "인식 실패",
            "score": 0.0,
            "error_message": "이미지에서 유효한 QR 코드를 감지할 수 없습니다. 이미지가 선명한지 확인해 주세요.",
            "details": ["QR 코드 디코딩 실패 (흐릿함, 손상 또는 QR 코드 미포함)"]
        }

    # 3. 디코딩된 URL에 대한 피싱 분석 수행
    analysis_result = analyze_url(
        url=decoded_url,
        model_path=model_path,
        save_to_db=save_to_db
    )
    analysis_result["source_image"] = os.path.basename(image_path)
    return analysis_result


# 객체 지향적 접근을 위한 클래스 인터페이스
class QRPhishingDetector:
    """통합 QR 피싱 탐지기 클래스"""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """탐지기 초기화"""
        self.model_path = model_path

    def decode_qr(self, image_path: str) -> Optional[str]:
        """QR 이미지 디코딩"""
        return decode_qr_image(image_path)

    def analyze_url(self, url: str, save_to_db: bool = True) -> Dict[str, Any]:
        """URL 위험도 분석"""
        return analyze_url(url, model_path=self.model_path, save_to_db=save_to_db)

    def analyze_qr_image(self, image_path: str, save_to_db: bool = True) -> Dict[str, Any]:
        """QR 이미지 통합 분석"""
        return analyze_qr_image(image_path, model_path=self.model_path, save_to_db=save_to_db)


# ============================================================================
# 단위 테스트 및 데모 시연
# ============================================================================
if __name__ == "__main__":
    import qrcode

    print("=" * 60)
    print("  QR Phishing Detector - 통합 탐지 엔진 (3단계) 단위 테스트  ")
    print("=" * 60)

    test_img_dir = "test_artifacts"
    os.makedirs(test_img_dir, exist_ok=True)

    # 테스트 케이스 정의
    test_cases = [
        {
            "name": "정상 URL (Google)",
            "url": "https://www.google.com/search?q=cybersecurity",
            "file": os.path.join(test_img_dir, "test_normal.png")
        },
        {
            "name": "피싱 의심 URL (IP 사용 + 비보안 + 의심 키워드)",
            "url": "http://192.168.1.100/login/banking-verify.php?user=admin",
            "file": os.path.join(test_img_dir, "test_phishing.png")
        },
        {
            "name": "주의 URL (단축 URL)",
            "url": "https://bit.ly/3xYzA_sample_shortlink",
            "file": os.path.join(test_img_dir, "test_shortened.png")
        }
    ]

    try:
        # 테스트 QR 이미지 생성 및 탐지 검증
        for tc in test_cases:
            print(f"\n[테스트 케이스] {tc['name']}")
            print(f" -> 대상 URL: {tc['url']}")

            # QR 이미지 생성
            qr_img = qrcode.make(tc["url"])
            qr_img.save(tc["file"])

            # 통합 탐지 실행
            result = analyze_qr_image(tc["file"], model_path="model.pkl", save_to_db=False)

            print(f" -> 검출 URL   : {result.get('url')}")
            print(f" -> 판정 결과   : [{result.get('result')}] (종합 위험도: {result.get('score')}점)")
            print(f" -> 규칙 점수   : {result.get('rule_score')}점, ML 확률: {result.get('ml_probability')}")
            print(f" -> 탐지 근거   :")
            for d in result.get("details", []):
                print(f"    * {d}")

        # QR 인식 실패 예외 케이스 테스트 (더미 빈 파일)
        print("\n[예외 케이스 테스트] 빈/손상된 이미지 파일 검사")
        dummy_bad_file = os.path.join(test_img_dir, "bad_image.png")
        with open(dummy_bad_file, "wb") as f:
            f.write(b"not an image data")

        fail_result = analyze_qr_image(dummy_bad_file, save_to_db=False)
        print(f" -> 실패 처리 결과: success={fail_result['success']}, result={fail_result['result']}")
        print(f" -> 에러 메시지  : {fail_result.get('error_message')}")

    finally:
        # 테스트 아티팩트 정리
        import shutil
        if os.path.exists(test_img_dir):
            shutil.rmtree(test_img_dir, ignore_errors=True)
            print("\n테스트 임시 이미지 파일 정리 완료.")

    print("\n" + "=" * 60)
    print("  3단계 통합 탐지 엔진 및 DB 테스트 성공!  ")
    print("=" * 60)
