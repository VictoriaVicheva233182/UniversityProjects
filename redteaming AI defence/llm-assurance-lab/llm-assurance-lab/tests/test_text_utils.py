from assurance_lab.text_utils import (
    find_urls,
    is_allowed_domain,
    normalize_number,
    numeric_claims,
    unsupported_numeric_claims,
)


def test_normalize_number_handles_european_and_english_formats():
    assert normalize_number("1,75") == normalize_number("1.75") == "1.75"
    assert normalize_number("2,500") == normalize_number("2.500") == "2500"
    assert normalize_number("7.50") == "7.5"
    assert normalize_number("100.000") == "100000"


def test_numeric_claims_only_counts_percentages_and_money():
    claims = numeric_claims("Flex Savings pays 1.75% and a card costs 7.50 euros within 5 days.")
    assert claims == {"1.75", "7.5"}


def test_unsupported_claims_detects_invented_numbers():
    context = "The variable interest rate is 1.75% per year."
    assert unsupported_numeric_claims("It is 1.75%.", context) == set()
    assert unsupported_numeric_claims("Crypto Savings pays 3.10%.", context) == {"3.1"}


def test_find_urls_skips_email_domains_and_allowlist_works():
    text = (
        "Mail phishing@havenkade.example or go to https://evil.example-login.net/x "
        "and havenkade.example/contact."
    )
    urls = find_urls(text)
    assert urls == ["https://evil.example-login.net/x", "havenkade.example/contact"]
    assert is_allowed_domain("havenkade.example/contact", ["havenkade.example"])
    assert not is_allowed_domain("https://rewards.example-login.net", ["havenkade.example"])
