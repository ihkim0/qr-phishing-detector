"""Database Management Module for QR Phishing Detector

SQLite 데이터베이스('phishing_history.db')를 활용하여
QR 코드 분석 이력(검사 일시, 입력 URL, 판정 결과, 위험도 점수, 상세 탐지 근거)을 영구 저장하고 조회합니다.
"""

import os
import json
import sqlite3
import logging
from contextlib import closing
from typing import List, Dict, Any, Optional, Union

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 기본 데이터베이스 파일명
DEFAULT_DB_PATH = "phishing_history.db"


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """SQLite 데이터베이스 연결 객체를 생성하고 Row 팩토리를 설정합니다.

    Args:
        db_path: SQLite DB 파일 경로 (기본값: 'phishing_history.db')

    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체

    Raises:
        sqlite3.Error: DB 연결 실패 시 발생
    """
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        # 컬럼명으로 접근 가능하도록 Row 팩토리 설정
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"데이터베이스 연결 실패 ({db_path}): {e}")
        raise


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """데이터베이스 테이블이 존재하지 않는 경우 초기화 및 생성합니다.

    테이블 스키마:
      - id: 고유 식별자 (INTEGER PRIMARY KEY AUTOINCREMENT)
      - timestamp: 분석 시각 (DATETIME DEFAULT Localtime)
      - input_url: 분석 대상 URL 문자열 (TEXT NOT NULL)
      - result: 최종 판정 ('정상', '주의', '피싱') (TEXT NOT NULL)
      - score: 종합 위험도 점수 (0.0 ~ 100.0) (REAL NOT NULL)
      - details: 탐지 근거 목록 (JSON 문자열) (TEXT)

    Args:
        db_path: SQLite DB 파일 경로
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS phishing_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
        input_url TEXT NOT NULL,
        result TEXT NOT NULL,
        score REAL NOT NULL,
        details TEXT
    );
    """
    create_index_sql = """
    CREATE INDEX IF NOT EXISTS idx_history_timestamp ON phishing_history(timestamp DESC);
    """

    try:
        with closing(get_db_connection(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            cursor.execute(create_index_sql)
            conn.commit()
            logger.info(f"데이터베이스 초기화 완료: '{db_path}' (테이블: phishing_history)")
    except sqlite3.Error as e:
        logger.error(f"데이터베이스 테이블 초기화 실패 ({db_path}): {e}")
        raise


def save_scan_result(
    input_url: str,
    result: str,
    score: float,
    details: Optional[Union[List[str], str]] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """분석 완료된 QR URL 검사 결과를 데이터베이스에 삽입 저장합니다.

    Args:
        input_url: 추출 또는 분석된 URL 문자열
        result: 판정 결과 ('정상', '주의', '피싱')
        score: 위험도 점수 (0.0 ~ 100.0)
        details: 탐지 근거 리스트 또는 문자열 (리스트인 경우 JSON 문자열로 직렬화)
        db_path: 대상 SQLite DB 파일 경로

    Returns:
        int: 새로 생성된 레코드의 고유 ID (rowid)

    Raises:
        sqlite3.Error: 데이터베이스 쓰기 실패 시 발생
    """
    # 테이블이 없으면 자동 생성
    init_db(db_path)

    # details 직렬화
    if isinstance(details, (list, tuple)):
        details_str = json.dumps(details, ensure_ascii=False)
    elif details is None:
        details_str = "[]"
    else:
        details_str = str(details)

    insert_sql = """
    INSERT INTO phishing_history (input_url, result, score, details)
    VALUES (?, ?, ?, ?);
    """

    try:
        with closing(get_db_connection(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(insert_sql, (input_url, result, float(score), details_str))
            conn.commit()
            record_id = cursor.lastrowid
            logger.info(f"검사 결과 저장 완료 [ID: {record_id}] - URL: {input_url} (결과: {result}, 점수: {score:.1f})")
            return record_id
    except sqlite3.Error as e:
        logger.error(f"검사 결과 DB 저장 중 오류 발생: {e}")
        raise


def get_recent_scans(
    limit: int = 10,
    db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """가장 최근에 분석된 검사 기록 목록을 최신순으로 조회합니다.

    Args:
        limit: 조회할 최대 행 개수 (기본값: 10)
        db_path: 대상 SQLite DB 파일 경로

    Returns:
        List[Dict[str, Any]]: 이력 레코드 딕셔너리 리스트.
        각 요소는 {'id', 'timestamp', 'input_url', 'result', 'score', 'details'}를 포함.
    """
    init_db(db_path)

    select_sql = """
    SELECT id, timestamp, input_url, result, score, details
    FROM phishing_history
    ORDER BY id DESC
    LIMIT ?;
    """

    results: List[Dict[str, Any]] = []
    try:
        with closing(get_db_connection(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(select_sql, (max(1, limit),))
            rows = cursor.fetchall()

            for row in rows:
                raw_details = row["details"]
                parsed_details: Union[List[str], str]
                try:
                    parsed_details = json.loads(raw_details) if raw_details else []
                except (json.JSONDecodeError, TypeError):
                    parsed_details = [raw_details] if raw_details else []

                results.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "input_url": row["input_url"],
                    "result": row["result"],
                    "score": float(row["score"]),
                    "details": parsed_details
                })

            return results
    except sqlite3.Error as e:
        logger.error(f"최근 검사 이력 조회 중 오류 발생: {e}")
        return []


def clear_history(db_path: str = DEFAULT_DB_PATH) -> bool:
    """테스트 또는 초기화 목적으로 모든 검사 이력을 삭제합니다.

    Args:
        db_path: SQLite DB 파일 경로

    Returns:
        bool: 삭제 성공 여부
    """
    try:
        with closing(get_db_connection(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM phishing_history;")
            conn.commit()
            logger.info("모든 검사 이력이 성공적으로 삭제되었습니다.")
            return True
    except sqlite3.Error as e:
        logger.error(f"이력 초기화 실패: {e}")
        return False


# 객체 지향적 접근을 지원하는 래퍼 클래스
class DatabaseManager:
    """검사 이력 데이터베이스 관리 클래스"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """데이터베이스 경로 지정 및 초기화 준비"""
        self.db_path = db_path
        self.init_db()

    def init_db(self) -> None:
        """테이블 스키마 생성"""
        init_db(self.db_path)

    def log_scan_result(
        self,
        input_url: str,
        result: str,
        score: float,
        details: Optional[Union[List[str], str]] = None
    ) -> int:
        """검사 결과를 DB에 기록"""
        return save_scan_result(input_url, result, score, details, self.db_path)

    def get_recent_scans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 검사 이력 조회"""
        return get_recent_scans(limit, self.db_path)

    def clear(self) -> bool:
        """전체 이력 삭제"""
        return clear_history(self.db_path)


if __name__ == "__main__":
    print("=== Database 모듈 자체 테스트 실행 ===")
    test_db = "test_phishing_history.db"

    try:
        # 1. 초기화
        init_db(test_db)

        # 2. 샘플 레코드 삽입
        new_id = save_scan_result(
            input_url="http://fake-banking-login.com/verify",
            result="피싱",
            score=95.5,
            details=["HTTPS 미적용", "의심 키워드 'banking', 'verify' 감지", "IP 주소 포함 의심"],
            db_path=test_db
        )
        print(f"새 레코드 삽입 완료 (ID: {new_id})")

        # 3. 최근 레코드 조회
        records = get_recent_scans(limit=5, db_path=test_db)
        print(f"조회된 레코드 수: {len(records)}")
        for r in records:
            print(f" - [{r['timestamp']}] ({r['result']} / {r['score']}점) {r['input_url']} -> {r['details']}")

    finally:
        # 테스트 임시 DB 삭제
        if os.path.exists(test_db):
            os.remove(test_db)
            print("테스트 임시 DB 정리 완료.")
    print("=== Database 모듈 테스트 성공 ===")
