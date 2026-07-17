"""Extraction schema for WildReceipt (North American English receipts).

store_name and date are text fields; price fields are dollar amounts.
line_items are the purchased product lines (description, quantity, price).
"""
from typing import List, Optional
from pydantic import BaseModel


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[str] = None
    price: Optional[str] = None


class ReceiptExtraction(BaseModel):
    store_name: Optional[str] = None
    date: Optional[str] = None
    subtotal_price: Optional[str] = None
    tax_price: Optional[str] = None
    tip_price: Optional[str] = None
    total_price: Optional[str] = None
    line_items: List[LineItem] = []


TARGET_FIELDS = ["store_name", "date", "subtotal_price", "tax_price",
                 "tip_price", "total_price"]
