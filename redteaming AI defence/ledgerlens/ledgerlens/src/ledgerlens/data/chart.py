"""Chart of accounts for Noordkade Logistics B.V. (fictional)."""

ACCOUNTS = {
    "1000": "Bank",
    "1100": "Accounts receivable",
    "1200": "Inventory",
    "1300": "Prepaid expenses",
    "1500": "Fixed assets",
    "1600": "Accumulated depreciation",
    "2000": "Accounts payable",
    "2100": "Accrued liabilities",
    "2200": "VAT payable",
    "2210": "VAT receivable",
    "2300": "Payroll liabilities",
    "4000": "Revenue",
    "5000": "Subcontracted transport",
    "6000": "Salaries and social charges",
    "6100": "Rent",
    "6200": "Vehicles and fuel",
    "6300": "IT and software",
    "6400": "Consulting and audit fees",
    "6500": "Marketing",
    "6600": "Travel",
    "6700": "Office and general",
    "6750": "Insurance",
    "6800": "Depreciation",
    "6900": "Bank charges",
    "7000": "Miscellaneous expenses",
}

VENDOR_CATEGORY_ACCOUNT = {
    "haulage subcontractor": "5000",
    "fuel": "6200",
    "vehicle maintenance": "6200",
    "vehicle lease": "6200",
    "it services": "6300",
    "software": "6300",
    "consulting": "6400",
    "marketing": "6500",
    "travel": "6600",
    "office supplies": "6700",
    "cleaning": "6700",
    "rent": "6100",
    "insurance": "6750",
}


def account_name(code: str) -> str:
    return ACCOUNTS.get(str(code), str(code))
