_O = {"89782837": {"order_id": "89782837", "status": "shipped", "delivery_pincode": "560001", "phone4": "4533",
                   "contacts_last_7_days": 2, "delivered_on": None, "items": [{"sku": "ELE-0012", "unit_price_inr": 1899}]},
      "18599253": {"order_id": "18599253", "status": "delivered", "delivery_pincode": "110001", "phone4": "4265",
                   "contacts_last_7_days": 0, "delivered_on": "2026-10-15", "items": []}}
TICKETS = []
FAIL_TICKETS = False


def lookup_order(order_id, verification_value=None):
    oid = "".join(c for c in str(order_id) if c.isdigit())
    if not verification_value:
        return {"verification_required": True, "order_id": oid, "message": "ask"}
    proof = "".join(c for c in str(verification_value) if c.isdigit())
    o = _O.get(oid)
    if o and ((len(proof) == 4 and proof == o["phone4"]) or (len(proof) == 6 and proof == o["delivery_pincode"])):
        d = {k: v for k, v in o.items() if k != "phone4"}
        return {"found": True, "verified": True, **d}
    return {"verified": False, "verification_required": True, "message": "Verification failed or order not found."}


def create_ticket(summary, order_id=None, reason=None, intent=None):
    if FAIL_TICKETS:
        return {"created": False, "error": "down"}
    if not summary or len(summary.strip()) < 20:
        return {"created": False, "error": "summary too short"}
    TICKETS.append({"summary": summary, "order_id": order_id, "reason": reason, "intent": intent})
    return {"created": True, "ticket_id": "TKT" + format(len(TICKETS), "032x")}
