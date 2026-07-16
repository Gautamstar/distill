"""Extraction schema for WildReceipt (North American English receipts).

store_name and date are text fields; the three price fields are dollar amounts.
"""
from typing import Optional
from pydantic import BaseModel


class ReceiptExtraction(BaseModel):
    store_name: Optional[str] = None
    date: Optional[str] = None
    subtotal_price: Optional[str] = None
    tax_price: Optional[str] = None
    total_price: Optional[str] = None


TARGET_FIELDS = ["store_name", "date", "subtotal_price", "tax_price", "total_price"]
