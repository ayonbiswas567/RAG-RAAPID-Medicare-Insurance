"""Mock tool tests: ICD-10 lookup, eligibility check, calculator, date difference."""

import pytest

from raapid_rag.tools.calculator import calculate, days_between
from raapid_rag.tools.eligibility import check_eligibility
from raapid_rag.tools.icd10 import icd10_lookup

# ------------------------------------------------------------------- ICD-10


def test_icd10_exact_code_lookup_is_case_and_whitespace_tolerant():
    r = icd10_lookup(" e11.22 ")
    assert r["found"] is True
    assert r["match_type"] == "code"
    assert r["code"] == "E11.22"
    assert "kidney" in r["description"].lower()
    assert r["hcc_category"] == "Diabetes with Chronic Complications"
    assert r["hcc_hierarchy"] == "Diabetes"


def test_icd10_term_search_returns_matching_codes():
    r = icd10_lookup("heart failure")
    assert r["found"] is True and r["match_type"] == "term"
    codes = {m["code"] for m in r["matches"]}
    assert {"I50.23", "I50.22"} <= codes
    assert all("heart failure" in m["description"].lower() for m in r["matches"])


def test_icd10_unknown_code_is_not_found():
    r = icd10_lookup("ZZZ.99")
    assert r["found"] is False
    assert r["matches"] == []


def test_icd10_codes_without_payment_hcc_say_so():
    r = icd10_lookup("I10")
    assert r["found"] is True
    assert r["hcc_category"] is None


def test_icd10_empty_query_is_an_error():
    assert "error" in icd10_lookup("   ")


def test_icd10_result_carries_illustrative_disclaimer():
    assert "illustrative" in icd10_lookup("E11.22")["note"].lower()


# -------------------------------------------------------------- eligibility


def test_eligibility_active_member_on_covered_date():
    r = check_eligibility("M-TEST-1001", "2025-03-14")
    assert r["status"] == "ACTIVE"
    assert r["plan"]
    assert r["coverage_start"] <= "2025-03-14" <= r["coverage_end"]


def test_eligibility_retro_terminated_member_is_inactive_after_termination():
    assert check_eligibility("M-TEST-1004", "2025-04-02")["status"] == "INACTIVE"
    assert check_eligibility("M-TEST-1004", "2025-03-01")["status"] == "ACTIVE"
    assert check_eligibility("M-TEST-1004", "2025-04-02")["termination_type"] == "retroactive"


def test_eligibility_unknown_member():
    r = check_eligibility("M-NOPE-0000", "2025-01-01")
    assert r["status"] == "NOT_FOUND"
    assert r["plan"] is None


def test_eligibility_bad_date_returns_error():
    assert "error" in check_eligibility("M-TEST-1001", "March 14th")


def test_eligibility_member_id_is_case_insensitive():
    assert check_eligibility("m-test-1001", "2025-03-14")["status"] == "ACTIVE"


# --------------------------------------------------------------- calculator


def test_calculate_arithmetic_with_whitelisted_functions():
    assert calculate("84 / (1.65 ** 2)")["result"] == pytest.approx(30.85, abs=0.01)
    assert calculate("round(84 / (1.65 ** 2), 1)")["result"] == 30.9
    assert calculate("max(3, 7) - min(1, 2)")["result"] == 6


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('dir')", "x + 1", "open('f')", "(1).__class__", "[1,2][0]", "'a' * 3", "2 ** 100000"],
)
def test_calculate_rejects_unsafe_or_unsupported_expressions(expr):
    assert "error" in calculate(expr)


def test_calculate_division_by_zero_is_an_error():
    assert "error" in calculate("1 / 0")


def test_calculate_syntax_error_is_an_error():
    assert "error" in calculate("2 +")


def test_days_between_iso_dates():
    r = days_between("2025-03-02", "2025-03-07")
    assert r["days"] == 5
    assert days_between("2025-03-07", "2025-03-02")["days"] == -5


def test_days_between_bad_date_is_an_error():
    assert "error" in days_between("2025-03-02", "yesterday")
