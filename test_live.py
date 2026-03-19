"""Quick live connectivity test — does NOT place any trades."""

from src.price_oracle import get_btc_price
from src.market_reader import discover_active_btc_5m_markets

print("=" * 60)
print("  POLYMARKET BOT — CONNECTIVITY TEST")
print("=" * 60)

print("\n1) BTC Price Oracle:")
result = get_btc_price()
for src, price in result["sources"].items():
    print(f"   {src}: ${price:,.2f}")
if result["median"]:
    print(f"   --> Median: ${result['median']:,.2f}")
print(f"   Sources responding: {result['count']}/4")

print("\n2) Active BTC 5-Min Markets:")
rounds = discover_active_btc_5m_markets()
if rounds:
    for r in rounds[:5]:
        print(f"\n   {r.question}")
        print(f"     Slug:       {r.event_slug}")
        print(f"     Ends in:    {r.seconds_remaining:.0f}s")
        print(f"     Up price:   ${r.up_price:.3f}")
        print(f"     Down price: ${r.down_price:.3f}")
        print(f"     Up token:   {r.up_token_id[:24]}...")
        print(f"     Down token: {r.down_token_id[:24]}...")
else:
    print("   No active rounds found")

ok = result["count"] >= 1 and len(rounds) >= 1
print(f"\n{'ALL SYSTEMS OK' if ok else 'WARNING: Some checks failed'}")
print("=" * 60)
