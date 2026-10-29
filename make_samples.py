"""QR Code Sample Generator Script

이 스크립트는 QR 피싱 탐지 시스템 테스트를 위해
정상 URL 2개와 피싱 의심 URL 2개의 QR 코드 이미지를 생성하여 'samples/' 폴더에 저장합니다.
"""

import os
import sys
import logging
from typing import List, Dict

import qrcode

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 샘플 QR 코드 정의
SAMPLES: List[Dict[str, str]] = [
    {
        "filename": "normal_1.png",
        "category": "정상 URL (공식 글로벌 서비스)",
        "url": "https://www.google.com",
        "description": "보안 프로토콜(HTTPS) 및 정규 도메인을 사용하는 가장 안전한 링크"
    },
    {
        "filename": "normal_2.png",
        "category": "정상 URL (교육/공공기관)",
        "url": "https://www.suwon.ac.kr",
        "description": "국내 교육기관 도메인(.ac.kr) 및 HTTPS가 적용된 정상 웹사이트"
    },
    {
        "filename": "phishing_1.png",
        "category": "피싱 의심 1 (IP 주소 + 비보안 + 의심 파라미터)",
        "url": "http://192.168.1.100/login?secure=update&token=xyz123abc7890",
        "description": "호스트명에 도메인 대신 사설 IP 주소 사용, 비암호화 HTTP 및 계정 갱신 유도 키워드 포함"
    },
    {
        "filename": "phishing_2.png",
        "category": "피싱 의심 2 (포털 사이트 사칭 + 다중 서브도메인)",
        "url": "http://login.naver.com.phishing-hack-site.kr/auth/signin",
        "description": "네이버 로그인(login.naver.com) 사칭 및 다중 서브도메인, HTTP 연결을 악용한 계정 탈취 링크"
    }
]


def generate_qr_samples(output_dir: str = "samples") -> List[str]:
    """정의된 샘플 목록에 따라 QR 코드 이미지를 생성하고 지정된 디렉토리에 저장합니다.

    Args:
        output_dir: QR 이미지들을 저장할 디렉토리 경로 (기본값: 'samples')

    Returns:
        List[str]: 생성 완료된 파일의 상대 경로 목록
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files: List[str] = []

    logger.info(f"샘플 QR 코드 생성 시작 (저장 경로: '{output_dir}/')...")

    for idx, item in enumerate(SAMPLES, start=1):
        target_path = os.path.join(output_dir, item["filename"])
        url = item["url"]

        try:
            # OpenCV QRCodeDetector 인식 호환성을 극대화하기 위해 오류 복원율 L 및 여백 설정
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)

            # 이미지 생성 및 저장 (PIL 기반)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(target_path)

            created_files.append(target_path)
            logger.info(
                f"[{idx}/4] {item['filename']} 생성 완료 | 분류: {item['category']} | URL: {url}"
            )

        except Exception as e:
            logger.error(f"{item['filename']} 생성 중 오류 발생 ({url}): {e}")

    logger.info("모든 샘플 QR 코드 생성이 완료되었습니다.")
    return created_files


def main() -> None:
    """메인 실행 함수"""
    print("=" * 65)
    print("  QR Phishing Detector - 테스트용 샘플 QR 이미지 생성기  ")
    print("=" * 65)

    created = generate_qr_samples("samples")

    print("\n[생성된 샘플 파일 목록]")
    for path in created:
        print(f" - {os.path.abspath(path)}")

    print("\n생성 완료! 웹 브라우저(http://localhost:5000)에서 위 이미지들을 드래그 앤 드롭하여 테스트할 수 있습니다.\n")


if __name__ == "__main__":
    main()
