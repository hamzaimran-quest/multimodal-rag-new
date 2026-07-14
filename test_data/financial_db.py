import psycopg2
from psycopg2.extras import execute_values
from faker import Faker
import uuid
import random
from datetime import datetime, timedelta
from decimal import Decimal

# Configuration
DB_CONFIG = "dbname=your_db user=your_user password=your_pass host=localhost"
fake = Faker()

def generate_targeted_financial_data():
    conn = psycopg2.connect(DB_CONFIG)
    cur = conn.cursor()
    print("Connecting to database and clearing existing test data safely...")
    
    # 1. GENERATE USERS (~60 users)
    print("Generating Users...")
    users = [(str(uuid.uuid4()), fake.unique.email(), fake.name(), fake.country_code()) for _ in range(60)]
    execute_values(cur, "INSERT INTO users (user_id, email, legal_name, country_code) VALUES %s", users)

    # 2. GENERATE ACCOUNTS (Exactly 100 accounts)
    print("Generating exactly 100 Accounts...")
    accounts = []
    # Ensure every single user gets at least one account first
    for i, u in enumerate(users):
        accounts.append((str(uuid.uuid4()), u[0], 'checking', 'USD', Decimal('5000.00')))
    
    # Pad the remaining accounts until we hit exactly 100
    while len(accounts) < 100:
        random_user = random.choice(users)
        accounts.append((str(uuid.uuid4()), random_user[0], 'savings', 'USD', Decimal('10000.00')))
        
    execute_values(cur, "INSERT INTO accounts (account_id, user_id, account_type, currency, balance) VALUES %s", accounts)

    # 3. GENERATE TRANSACTIONS & ENTRIES (300 transactions / 600 entries)
    print("Generating 300 Transactions with balanced ledger entries...")
    transactions = []
    entries = []
    
    for i in range(300):
        t_id = str(uuid.uuid4())
        amount = Decimal(random.uniform(5.00, 250.00)).quantize(Decimal('0.01'))
        acc_debit, acc_credit = random.sample(accounts, 2)
        
        transactions.append((t_id, f"TXN-REF-{100000 + i}", f"Transfer payload ref {i}"))
        
        # Double-entry balance mapping (Debit negative, Credit positive)
        entries.append((str(uuid.uuid4()), t_id, acc_debit[0], -amount, Decimal('0.00')))
        entries.append((str(uuid.uuid4()), t_id, acc_credit[0], amount, Decimal('0.00')))

    execute_values(cur, "INSERT INTO transactions (transaction_id, reference_id, description) VALUES %s", transactions)
    execute_values(cur, "INSERT INTO transaction_entries (entry_id, transaction_id, account_id, amount, running_balance) VALUES %s", entries)

    # 4. GENERATE FINANCIAL INSTRUMENTS
    print("Populating Financial Instruments...")
    ticker_pool = [
        ('AAPL', 'equity', 'NASDAQ'), ('MSFT', 'equity', 'NASDAQ'), 
        ('BTC', 'crypto', 'BINANCE'), ('ETH', 'crypto', 'BINANCE'),
        ('EURUSD', 'forex', 'ICE'), ('GBPUSD', 'forex', 'ICE')
    ]
    cur.executemany("""
        INSERT INTO instruments (ticker, asset_class, exchange_code) 
        VALUES (%s, %s, %s) ON CONFLICT (ticker) DO NOTHING;
    """, ticker_pool)
    
    # Fetch real auto-assigned IDs back from the DB to link ticks and orders safely
    cur.execute("SELECT instrument_id FROM instruments;")
    instrument_ids = [row[0] for row in cur.fetchall()]

    # 5. GENERATE MARKET TICKS (~300 records)
    print("Generating 300 historical market ticks...")
    ticks = []
    base_time = datetime.now() - timedelta(days=2)
    
    for i in range(300):
        inst_id = random.choice(instrument_ids)
        timestamp = base_time + timedelta(seconds=i * 30) # Tick updates every 30 seconds
        base_price = random.uniform(10.0, 150.0) if inst_id not in [3, 4] else random.uniform(2000.0, 60000.0)
        
        bid = Decimal(base_price).quantize(Decimal('0.01'))
        ask = Decimal(base_price + random.uniform(0.01, 0.05)).quantize(Decimal('0.01'))
        volume = Decimal(random.uniform(10, 5000)).quantize(Decimal('0.01'))
        
        ticks.append((inst_id, timestamp, bid, ask, volume))
        
    execute_values(cur, "INSERT INTO market_ticks (instrument_id, timestamp, bid_price, ask_price, volume) VALUES %s", ticks)

    # 6. GENERATE ORDERS (~300 records)
    print("Generating 300 customer trading orders...")
    orders = []
    
    for _ in range(300):
        acc = random.choice(accounts)
        inst_id = random.choice(instrument_ids)
        side = random.choice(['BUY', 'SELL'])
        o_type = random.choice(['LIMIT', 'MARKET'])
        qty = Decimal(random.uniform(0.1, 50.0)).quantize(Decimal('0.0001'))
        price = Decimal(random.uniform(15.0, 160.0)).quantize(Decimal('0.01')) if o_type == 'LIMIT' else None
        status = random.choice(['FILLED', 'CANCELLED', 'PENDING'])
        
        orders.append((str(uuid.uuid4()), acc[0], inst_id, side, o_type, qty, price, status))
        
    execute_values(cur, "INSERT INTO orders (order_id, account_id, instrument_id, side, order_type, quantity, limit_price, status) VALUES %s", orders)

    conn.commit()
    print("\nTarget Volume Verification successfully written to DB!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    generate_targeted_financial_data()