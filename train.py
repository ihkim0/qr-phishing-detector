"""URL Phishing Detection - Model Training Script

이 스크립트는 URL 문자열로부터 다차원 특징(Feature)을 추출하고,
Scikit-learn의 RandomForestClassifier를 이용해 피싱 여부를 판별하는 머신러닝 모델을 학습시킵니다.
학습 완료 후 모델 및 메타데이터는 'model.pkl'로 직렬화 저장됩니다.
"""

import os
import re
import sys
import argparse
import logging
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 특징(Feature) 이름 목록 정의 (Detector 모듈과 일치 유지)
FEATURE_NAMES: List[str] = [
    "url_length",             # 전체 URL 길이
    "hostname_length",        # 호스트명(도메인) 길이
    "path_length",            # 경로(Path) 길이
    "count_dots",             # 마침표('.') 개수
    "count_hyphens",          # 하이픈('-') 개수
    "count_at",               # 골뱅이('@') 개수 (피싱의 Basic Auth 난독화 패턴)
    "count_question",         # 물음표('?') 개수
    "count_equal",            # 등호('=') 개수
    "count_slash",            # 슬래시('/') 개수
    "count_percent",          # 퍼센트('%') 개수 (URL 인코딩 난독화)
    "count_digits",           # 숫자 총 개수
    "digit_ratio",            # URL 전체 대비 숫자 비율
    "has_ip_address",         # IP 주소 직접 사용 여부 (1: 포함, 0: 미포함)
    "is_https",               # HTTPS 보안 프로토콜 여부 (1: HTTPS, 0: HTTP 등)
    "num_subdomains",         # 서브도메인 깊이 (점 개수 기반 계산)
    "suspicious_keyword_cnt", # 피싱 의심 키워드 포함 개수
    "is_shortened_url"        # 주요 단축 URL 서비스 사용 여부 (1: 단축 URL, 0: 일반)
]

# 피싱에서 자주 악용되는 의심 키워드 목록
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

# IPv4 정규식 패턴 (호스트명에 IP 주소 직접 사용 여부 검출)
IP_PATTERN = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def extract_url_features(url: str) -> Dict[str, float]:
    """단일 URL 문자열로부터 머신러닝 학습에 필요한 수치형 특징(Feature) 벡터를 추출합니다.

    Args:
        url: 분석 대상 URL 문자열 (예: 'https://example.com/login')

    Returns:
        특징명(키)과 해당 수치값(값)으로 구성된 딕셔너리
    """
    if not isinstance(url, str) or not url.strip():
        # 빈 문자열이거나 유효하지 않은 입력인 경우 기본 0값 반환
        return {name: 0.0 for name in FEATURE_NAMES}

    url_str = url.strip()

    # 스키마(프로토콜) 누락 시 기본 http:// 부여하여 urlparse 정상 동작 유도
    if not url_str.startswith(("http://", "https://")):
        parsed = urlparse("http://" + url_str)
        is_https = 0.0
    else:
        parsed = urlparse(url_str)
        is_https = 1.0 if parsed.scheme.lower() == "https" else 0.0

    hostname = parsed.netloc.split(":")[0].lower() if parsed.netloc else ""
    path = parsed.path

    # 1. 길이 관련 특징
    url_length = float(len(url_str))
    hostname_length = float(len(hostname))
    path_length = float(len(path))

    # 2. 특수문자 개수
    count_dots = float(url_str.count("."))
    count_hyphens = float(url_str.count("-"))
    count_at = float(url_str.count("@"))
    count_question = float(url_str.count("?"))
    count_equal = float(url_str.count("="))
    count_slash = float(url_str.count("/"))
    count_percent = float(url_str.count("%"))

    # 3. 숫자 개수 및 비율
    digits = sum(c.isdigit() for c in url_str)
    count_digits = float(digits)
    digit_ratio = float(digits / len(url_str)) if url_length > 0 else 0.0

    # 4. IP 주소 사용 여부 검출
    has_ip = 1.0 if IP_PATTERN.match(hostname) else 0.0

    # 5. 서브도메인 개수 계산 (도메인의 점 개수 기반)
    # 일반적인 도메인 (예: naver.com)은 점이 1개이므로 서브도메인은 0개
    subdomain_parts = hostname.split(".")
    num_subdomains = float(max(0, len(subdomain_parts) - 2)) if len(subdomain_parts) > 1 else 0.0

    # 6. 피싱 의심 키워드 카운트
    url_lower = url_str.lower()
    keyword_cnt = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    suspicious_keyword_cnt = float(keyword_cnt)

    # 7. 단축 URL 서비스 이용 여부
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


def extract_features_batch(urls: List[str]) -> pd.DataFrame:
    """복수의 URL 리스트로부터 특징 행렬(DataFrame)을 생성합니다.

    Args:
        urls: URL 문자열 리스트

    Returns:
        FEATURE_NAMES 순서의 컬럼을 가진 pandas DataFrame
    """
    feature_list = [extract_url_features(url) for url in urls]
    df = pd.DataFrame(feature_list)
    return df[FEATURE_NAMES]


def get_dummy_dataset() -> pd.DataFrame:
    """CSV 데이터셋이 없을 때 모델 학습 및 테스트를 위해 사전 정의된 현실적인 더미 데이터셋을 생성합니다.

    Returns:
        'url' 및 'label' 컬럼을 포함하는 DataFrame (0: 정상, 1: 피싱)
    """
    # 정상 URL 샘플 (label: 0)
    legitimate_urls = [
        "https://www.google.com",
        "https://www.google.com/search?q=capstone+design",
        "https://www.naver.com",
        "https://news.naver.com/section/105",
        "https://mail.naver.com",
        "https://www.daum.net",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://github.com/scikit-learn/scikit-learn",
        "https://github.com/opencv/opencv",
        "https://www.wikipedia.org",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://www.amazon.com",
        "https://www.apple.com/kr",
        "https://www.microsoft.com/ko-kr",
        "https://www.python.org/downloads/",
        "https://docs.python.org/3/library/urllib.parse.html",
        "https://scikit-learn.org/stable/modules/ensemble.html",
        "https://stackoverflow.com/questions/tagged/machine-learning",
        "https://www.acmicpc.net/problem/1000",
        "https://velog.io/@developer/weekly-retrospective",
        "https://tistory.com",
        "https://www.kaggle.com/datasets",
        "https://opencv.org/releases/",
        "https://flask.palletsprojects.com/en/3.0.x/quickstart/",
        "https://pypi.org/project/opencv-python/",
        "https://ko.aliexpress.com",
        "https://www.coupang.com",
        "https://www.inflearn.com",
        "https://programmers.co.kr",
        "https://github.com/explore"
    ]

    # 피싱 / 악성 URL 샘플 (label: 1)
    phishing_urls = [
        "http://192.168.0.1/login/verify.html",
        "http://185.220.101.5/banking/account-update",
        "http://172.56.21.99/signin/naver-auth/index.php",
        "http://secure-login-naver.com-verify.account-security.net/signin",
        "http://paypal.com.account-verification-notice.xyz/login?id=user",
        "http://apple-id-reset-alert.info/verify?token=928471209384",
        "http://www.login.google.com.user-auth-check.top/update",
        "http://netflix-account-suspended-update.site/payment/confirm",
        "http://kakao-pay-security-center.club/auth/login.php",
        "http://kakaotalk-login-verify-service.cc/login.php?user=alert",
        "http://shinhan-bank-safety-cert.tk/cert/login",
        "http://kb-star-banking-support.ml/verify-account.asp?code=8831",
        "http://bit.ly/3xXyZ9_fake_login_phish",
        "http://tinyurl.com/account-suspended-confirm-now",
        "http://woori-bank-security-center.ga/login.html?session=abc1234",
        "http://instagram-verify-badge-request.xyz/login/verify?ref=alert",
        "http://security-update-google-warning.top/account/protect",
        "http://amazon-security-order-verification.cc/order/confirm.php",
        "http://microsoft-account-suspicious-login.info/verify",
        "http://104.244.42.1/update-profile/index.php?user=test",
        "http://facebook-security-checkup-notice.site/checkpoint",
        "http://bank-account-re-login-verify.top/auth?token=93821039",
        "http://naver-safety-login-center.xyz/nidlogin.php?mode=number",
        "http://daum-mail-account-expired.club/login?target=mail",
        "http://cutt.ly/bank-suspended-login",
        "http://update-wallet-crypto-security.info/metamask/verify.html",
        "http://apple-icloud-findmy-alert-lock.site/unlock/auth",
        "http://t.co/fakephishingscamurl",
        "http://safety-check-google-auth2.com/login?service=mail",
        "http://198.51.100.23/verify-account/billing/signin.html"
    ]

    data = []
    for u in legitimate_urls:
        data.append({"url": u, "label": 0})
    for u in phishing_urls:
        data.append({"url": u, "label": 1})

    df = pd.DataFrame(data)
    # 데이터 순서 무작위 셔플
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """CSV 파일 경로를 받아 데이터셋을 로드하거나, 없을 경우 더미 데이터를 생성합니다.

    Args:
        csv_path: CSV 파일 경로 (선택적)

    Returns:
        'url' 및 'label' 컬럼을 갖춘 pandas DataFrame
    """
    if csv_path and os.path.exists(csv_path):
        logger.info(f"외부 CSV 데이터셋 로드 중: {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            # 유연한 컬럼명 매핑
            col_map = {}
            for col in df.columns:
                lower = col.strip().lower()
                if lower in ["url", "urls", "link", "website"]:
                    col_map[col] = "url"
                elif lower in ["label", "target", "class", "result", "status", "phishing"]:
                    col_map[col] = "label"

            df = df.rename(columns=col_map)

            if "url" not in df.columns or "label" not in df.columns:
                raise ValueError(
                    f"CSV 파일에 'url'과 'label'에 대응하는 컬럼이 없습니다. (현재 컬럼: {list(df.columns)})"
                )

            # 라벨이 문자열(예: 'phishing'/'legitimate' 또는 'bad'/'good')인 경우 수치형 변환
            if df["label"].dtype == object:
                mapping = {
                    "phishing": 1, "bad": 1, "malicious": 1, "1": 1,
                    "legitimate": 0, "good": 0, "benign": 0, "0": 0
                }
                df["label"] = df["label"].astype(str).str.strip().str.lower().map(mapping)

            # 결측치 제거
            df = df.dropna(subset=["url", "label"])
            df["label"] = df["label"].astype(int)

            logger.info(f"성공적으로 {len(df)}개 레코드를 로드했습니다. (정상: {(df['label'] == 0).sum()}, 피싱: {(df['label'] == 1).sum()})")
            return df[["url", "label"]]

        except Exception as e:
            logger.error(f"CSV 파일 로드 실패: {e}. 더미 데이터셋으로 대체합니다.")
            return get_dummy_dataset()

    logger.info("지정된 데이터셋이 없거나 경로가 유효하지 않아 내장 더미 데이터셋(60개 샘플)을 사용합니다.")
    return get_dummy_dataset()


def train_model(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """데이터프레임으로부터 특징을 추출하고 RandomForest 분류기를 학습합니다.

    Args:
        df: 'url'과 'label' 컬럼이 포함된 DataFrame
        test_size: 테스트 데이터셋 비율 (기본: 0.2)
        random_state: 재현성을 위한 난수 시드 (기본: 42)

    Returns:
        (학습된 RandomForestClassifier 모델, 평가 메트릭 딕셔너리)
    """
    logger.info("URL 특징 추출 진행 중...")
    X = extract_features_batch(df["url"].tolist())
    y = df["label"].values

    # 학습 및 테스트 세트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"데이터셋 분할 완료 - 학습 세트: {len(X_train)}개, 테스트 세트: {len(X_test)}개")

    # RandomForest 모델 초기화 및 학습
    # 캡스톤 환경에서 빠르고 과적합을 방지하기 위한 하이퍼파라미터 설정
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )

    logger.info("RandomForestClassifier 학습 시작...")
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info("=" * 50)
    logger.info("모델 학습 및 평가 결과:")
    logger.info(f" - 정확도 (Accuracy)  : {acc * 100:.2f}%")
    logger.info(f" - 정밀도 (Precision) : {prec * 100:.2f}%")
    logger.info(f" - 재현율 (Recall)    : {rec * 100:.2f}%")
    logger.info(f" - F1-Score           : {f1:.4f}")
    logger.info("=" * 50)
    logger.info("\n[혼동 행렬 (Confusion Matrix)]\n" + str(confusion_matrix(y_test, y_pred)))
    logger.info("\n[분류 리포트 (Classification Report)]\n" + classification_report(y_test, y_pred, target_names=["정상(0)", "피싱(1)"]))

    # 특징 중요도(Feature Importance) 출력
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("[주요 특징 중요도 Top 5]")
    for i in range(min(5, len(FEATURE_NAMES))):
        idx = sorted_idx[i]
        logger.info(f"  {i+1}. {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }

    return model, metrics


def save_model(
    model: RandomForestClassifier,
    save_path: str = "model.pkl",
    metrics: Optional[Dict[str, float]] = None
) -> None:
    """학습된 모델과 관련 메타데이터를 파일로 저장합니다.

    Args:
        model: 학습된 Scikit-learn 모델
        save_path: 저장할 파일 경로 (기본: 'model.pkl')
        metrics: 모델 평가 메트릭 (선택적)
    """
    # 탐지 엔진(detector.py)에서 원활하게 로드할 수 있도록
    # 모델 객체뿐만 아니라 사용된 특징 목록(FEATURE_NAMES)과 메타데이터를 함께 딕셔너리로 패키징
    package = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics or {},
        "version": "1.0.0"
    }

    try:
        joblib.dump(package, save_path)
        logger.info(f"모델 및 메타데이터가 성공적으로 저장되었습니다: '{save_path}'")
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="QR 피싱 탐지 시스템 - URL 피싱 분류 ML 모델 학습 스크립트"
    )
    parser.add_argument(
        "--dataset", "--csv",
        type=str,
        default=None,
        help="학습에 사용할 CSV 데이터셋 경로 ('url', 'label' 컬럼 필요). 지정하지 않을 경우 내장 더미 데이터셋 사용."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="model.pkl",
        help="학습된 모델 저장 파일명 (기본값: 'model.pkl')"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="테스트 데이터 분할 비율 (기본값: 0.2)"
    )
    return parser.parse_args()


def main() -> None:
    """ML 모델 학습 메인 파이프라인 함수"""
    args = parse_arguments()

    print("=" * 60)
    print("  QR Phishing Detector - 머신러닝 모델 학습 (2단계)  ")
    print("=" * 60)

    # 1. 데이터셋 로드
    df = load_dataset(args.dataset)

    # 2. 모델 학습 및 평가
    model, metrics = train_model(df, test_size=args.test_size)

    # 3. 모델 직렬화 저장
    save_model(model, save_path=args.output, metrics=metrics)

    print("\n[완료] 머신러닝 모델 학습 및 저장이 성공적으로 완료되었습니다!")
    print(f"       저장 경로: {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
