# tests/test_pii.py
from contract_rag.security.pii import PIIType, detect_pii


def test_detects_email_and_ssn():
    text = "Contact jane.doe@acme.com or SSN 123-45-6789 for details."
    found = {m.type for m in detect_pii(text)}
    assert PIIType.EMAIL in found
    assert PIIType.SSN in found


def test_detects_ip_and_phone_and_credit_card():
    text = "Server 10.0.12.4, call (415) 555-0199, card 4111 1111 1111 1111."
    found = {m.type for m in detect_pii(text)}
    assert PIIType.IP in found
    assert PIIType.PHONE in found
    assert PIIType.CREDIT_CARD in found


def test_clean_contract_text_has_no_false_positives():
    text = "This Agreement is governed by the laws of the State of New York."
    assert detect_pii(text) == []


def test_types_filter_limits_scan():
    text = "jane@acme.com 123-45-6789"
    only_ssn = detect_pii(text, types=[PIIType.SSN])
    assert [m.type for m in only_ssn] == [PIIType.SSN]


def test_matches_sorted_by_start_offset():
    text = "ssn 123-45-6789 then email jane@acme.com"
    starts = [m.start for m in detect_pii(text)]
    assert starts == sorted(starts)


def test_no_false_positive_inside_a_longer_digit_run():
    # A 14-digit invoice number must not be matched as a phone.
    text = "Invoice No. 12345678901234 is due."
    phones = [m for m in detect_pii(text) if m.type == PIIType.PHONE]
    assert phones == []


def test_ip_section_reference_is_not_pii():
    # The S2-review over-redaction case: a dotted section number is not an IP.
    text = "As set out in Section 1.2.3.4 of this Agreement."
    assert [m for m in detect_pii(text) if m.type == PIIType.IP] == []


def test_ip_section_symbol_reference_is_not_pii():
    text = "See §1.2.3.4 below."
    assert [m for m in detect_pii(text) if m.type == PIIType.IP] == []


def test_real_ip_after_non_section_word_still_detected():
    text = "Server 10.0.12.4 hosts the data."
    ips = [m.value for m in detect_pii(text) if m.type == PIIType.IP]
    assert ips == ["10.0.12.4"]


def test_out_of_range_octet_is_not_ip():
    text = "build 999.1.2.3 shipped"
    assert [m for m in detect_pii(text) if m.type == PIIType.IP] == []


def test_dotted_quad_inside_longer_run_is_not_ip():
    # A 5-group version string must not yield an embedded fake IP.
    text = "version 1.2.3.4.5 released"
    assert [m for m in detect_pii(text) if m.type == PIIType.IP] == []
