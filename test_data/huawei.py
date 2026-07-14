import psycopg2
from psycopg2.extras import execute_values
from decimal import Decimal

DB_CONFIG = "dbname=huawei_report user=postgres password=secret_pass host=localhost port=5435"

def populate_mock_data():
    conn = psycopg2.connect(DB_CONFIG)
    cur = conn.cursor()
    
    # 1. Regions Population (Simulated Global Footprint)
    print("Populating Mock Regional Data...")
    regions_data = [
        ('Domestic Market', Decimal('590200.00'), Decimal('585100.00'), Decimal('0.87')),
        ('Europe & Middle East', Decimal('175400.00'), Decimal('162300.00'), Decimal('8.07')),
        ('Asia-Pacific Hub', Decimal('55100.00'), Decimal('48200.00'), Decimal('14.31')),
        ('Americas Division', Decimal('39800.00'), Decimal('38100.00'), Decimal('4.46')),
        ('Emerging International territories', Decimal('18100.00'), Decimal('19900.00'), Decimal('-9.05'))
    ]
    execute_values(cur, """
        INSERT INTO regions (region_name, revenue_2025_cny_million, revenue_2024_cny_million, yoy_growth_percentage) 
        VALUES %s""", regions_data)

    # 2. Business Segments Population (Simulated Operating Segments)
    print("Populating Mock Business Segment Performance...")
    segments_data = [
        ('Core Enterprise Networks', Decimal('388000.00'), Decimal('371000.00'), Decimal('4.58')),
        ('Hardware Devices & Consumer Tech', Decimal('351000.00'), Decimal('342000.00'), Decimal('2.63')),
        ('Enterprise Cloud Platforms', Decimal('35500.00'), Decimal('34100.00'), Decimal('4.11')),
        ('Green Energy & Digital Power', Decimal('79200.00'), Decimal('71400.00'), Decimal('10.92')),
        ('Smart Mobility Solutions', Decimal('42300.00'), Decimal('29800.00'), Decimal('41.95'))
    ]
    execute_values(cur, """
        INSERT INTO business_segments (segment_name, revenue_2025_cny_million, revenue_2024_cny_million, yoy_growth_percentage) 
        VALUES %s""", segments_data)

    # 3. Five-Year Financial Highlights Population (Simulated Historical Trends)
    print("Populating Mock Historical Performance...")
    financial_data = [
        (2025, Decimal('896000.00'), Decimal('98100.00'), Decimal('10.95'), Decimal('69500.00'), Decimal('129000.00'), Decimal('1350000.00'), Decimal('612000.00'), Decimal('54.66')),
        (2024, Decimal('855000.00'), Decimal('81000.00'), Decimal('9.47'), Decimal('64100.00'), Decimal('89500.00'), Decimal('1285000.00'), Decimal('552000.00'), Decimal('57.04')),
        (2023, Decimal('712000.00'), Decimal('101000.00'), Decimal('14.19'), Decimal('85000.00'), Decimal('71000.00'), Decimal('1250000.00'), Decimal('511000.00'), Decimal('59.12')),
        (2022, Decimal('638000.00'), Decimal('44000.00'), Decimal('6.90'), Decimal('37000.00'), Decimal('18500.00'), Decimal('1070000.00'), Decimal('441000.00'), Decimal('58.79')),
        (2021, Decimal('641000.00'), Decimal('119000.00'), Decimal('18.56'), Decimal('112000.00'), Decimal('61000.00'), Decimal('991000.00'), Decimal('419000.00'), Decimal('57.72'))
    ]
    execute_values(cur, """
        INSERT INTO financial_highlights (year, revenue_cny_million, operating_profit_cny_million, operating_margin_percentage, net_profit_cny_million, cash_flow_operating_cny_million, total_assets_cny_million, equity_cny_million, liability_ratio_percentage) 
        VALUES %s""", financial_data)

    # 4. Tech Ecosystem Ecosystem Population (Simulated Developer Traction)
    print("Populating Mock Ecosystem Developer Platform Metrics...")
    ecosystem_data = [
        ('NextGen OS Platform', 9500000, 15000, 320000),      
        ('Legacy Hardware Architecture', 3600000, 6200, 19500),          
        ('AI Accelerators Ecosystem', 4200000, 3100, 6900),            
        ('Distributed Compute Infrastructure', 10500000, 61000, 45000)     
    ]
    execute_values(cur, """
        INSERT INTO tech_ecosystems (platform_name, global_registered_developers, partner_count, compatible_solutions_incubated) 
        VALUES %s""", ecosystem_data)

    # 5. Operational Global Footprint Metrics (Simulated Reach Centers)
    print("Populating Mock Global Retail & Footprint Scales...")
    footprint_data = [
        ('Premium Experience Centers', 18, 5),
        ('Smart Integrated Retail Locations', 495, 12),
        ('Global Field Support Hubs', 3250, 72), 
        ('Network Points of Presence', 105, 175)          
    ]
    execute_values(cur, """
        INSERT INTO operational_footprint (metric_name, global_count, countries_covered) 
        VALUES %s""", footprint_data)

    # 6. Linking Many-to-Many Relationships (Mapping)
    print("Creating mock relational dependencies...")
    cur.execute("SELECT segment_id, segment_name FROM business_segments;")
    seg_map = {name: s_id for s_id, name in cur.fetchall()}
    
    cur.execute("SELECT ecosystem_id, platform_name FROM tech_ecosystems;")
    eco_map = {name: e_id for e_id, name in cur.fetchall()}

    mappings = [
        (seg_map['Hardware Devices & Consumer Tech'], eco_map['NextGen OS Platform']),
        (seg_map['Enterprise Cloud Platforms'], eco_map['Distributed Compute Infrastructure']),
        (seg_map['Core Enterprise Networks'], eco_map['Legacy Hardware Architecture']),
        (seg_map['Core Enterprise Networks'], eco_map['AI Accelerators Ecosystem'])
    ]
    execute_values(cur, "INSERT INTO segment_platform_mapping (segment_id, ecosystem_id) VALUES %s", mappings)

    conn.commit()
    print("Mock database populated successfully with randomized, non-contradictory metrics.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    populate_mock_data()