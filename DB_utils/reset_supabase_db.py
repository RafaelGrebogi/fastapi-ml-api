import json
import sys
import os
from datetime import date

# Add the root folder to sys.path so you can import supabase_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from supabase_client import supabase

def reset_supabase_db():
    # ------------------------------
    # Clear Tables (preserve user id=1 and device ids 1 & 2)
    # ------------------------------
    supabase.table("results").delete().neq("id", -1).execute()
    supabase.table("service").delete().neq("id", -1).execute()
    supabase.table("company").delete().neq("id", -1).execute()
    supabase.table("users_ext").delete().neq("id", -1).execute()
    supabase.table("users").delete().neq("id", 1).execute()
    supabase.table("device").delete().gt("id", 2).execute()  

    # ------------------------------
    # Insert Users
    # ------------------------------
    users_data = [
        {"name": "Albert Einstein", "username": "einstein", "is_admin": True, "is_active": True, "email": "einstein@example.com", "phone": "+61411111111", "date_birth": "1950-03-14"},
        {"name": "Marie Curie", "username": "curie", "is_admin": False, "is_active": True, "email": "curie@example.com", "phone": "+61411111112", "date_birth": "1975-11-07"},
        {"name": "Isaac Newton", "username": "newton", "is_admin": False, "is_active": True, "email": "newton@example.com", "phone": "+61411111113", "date_birth": "1985-01-04"},
        {"name": "Niels Bohr", "username": "bohr", "is_admin": False, "is_active": False, "email": "bohr@example.com", "phone": "+61411111114", "date_birth": "1992-10-07"},
        {"name": "Galileo Galilei", "username": "galileo", "is_admin": False, "is_active": True, "email": "galileo@example.com", "phone": "+61411111115", "date_birth": "1990-02-15"},
        {"name": "Richard Feynman", "username": "feynman", "is_admin": False, "is_active": True, "email": "feynman@example.com", "phone": "+61411111116", "date_birth": "1968-05-11"},
        {"name": "Stephen Hawking", "username": "hawking", "is_admin": True, "is_active": True, "email": "hawking@example.com", "phone": "+61411111117", "date_birth": "1960-01-08"},
        {"name": "Paul Dirac", "username": "dirac", "is_admin": False, "is_active": False, "email": "dirac@example.com", "phone": "+61411111118", "date_birth": "1980-08-08"},
        {"name": "Enrico Fermi", "username": "fermi", "is_admin": False, "is_active": True, "email": "fermi@example.com", "phone": "+61411111119", "date_birth": "1970-09-29"},
        {"name": "James Clerk Maxwell", "username": "maxwell", "is_admin": False, "is_active": True, "email": "maxwell@example.com", "phone": "+61411111120", "date_birth": "1972-06-13"},
        {"name": "Lise Meitner", "username": "meitner", "is_admin": False, "is_active": True, "email": "meitner@example.com", "phone": "+61411111121", "date_birth": "1980-11-17"},
        {"name": "Erwin Schrödinger", "username": "schrodinger", "is_admin": False, "is_active": True, "email": "schrodinger@example.com", "phone": "+61411111122", "date_birth": "1978-08-12"}
    ]
    inserted_users = supabase.table("users").insert(users_data).execute().data
    user_id_map = {u["username"]: u["id"] for u in inserted_users}
    user_id_map["admin"] = 1

    # ------------------------------
    # Insert Companies
    # ------------------------------
    companies_data = [
        {"name": "Greyhound Dynamics", "owner_id": user_id_map["curie"], "email": "greyhound1@corp.com", "phone": "+61211111111", "address": "1 Street NSW", "is_active": True},
        {"name": "BorderCollie Systems", "owner_id": user_id_map["newton"], "email": "bcollie@corp.com", "phone": "+61211111112", "address": "2 Street NSW", "is_active": True},
        {"name": "Whippet AI", "owner_id": user_id_map["bohr"], "email": "whippet@corp.com", "phone": "+61211111113", "address": "3 Street NSW", "is_active": True},
        {"name": "Poodle Analytics", "owner_id": user_id_map["galileo"], "email": "poodle@corp.com", "phone": "+61211111114", "address": "4 Street NSW", "is_active": True},
        {"name": "BeagleTech", "owner_id": user_id_map["feynman"], "email": "beagle@corp.com", "phone": "+61211111115", "address": "5 Street NSW", "is_active": False},
        {"name": "RafAI Systems", "owner_id": 1, "email": "rafai@corp.com", "phone": "+61211111116", "address": "6 Street NSW", "is_active": True}
    ]
    inserted_companies = supabase.table("company").insert(companies_data).execute().data
    company_ids = [c["id"] for c in inserted_companies]
    company_map = {c["name"]: c["id"] for c in inserted_companies}
    rafai_id = company_map["RafAI Systems"]

    # ------------------------------
    # Insert Dummy Devices
    # ------------------------------
    dummy_devices = []
    for i in range(8):
        assigned_company = company_ids[i % len(company_ids)]
        dummy_devices.append({
            "name": f"Test_Device_{i + 1}",
            "serial_number": f"TEST-SN-{i + 1:03}",
            "is_available": True,
            "in_use": False,
            "notes": f"Dummy device {i + 1}",
            "company_id": assigned_company
        })
    inserted_devices = supabase.table("device").insert(dummy_devices).execute().data

    device_pool_by_company = {}
    for device in inserted_devices:
        company_id = device["company_id"]
        device_pool_by_company.setdefault(company_id, []).append(device["id"])
    device_pool_by_company.setdefault(rafai_id, []).extend([1, 2])  # manual real boards

    # ------------------------------
    # Insert Services
    # ------------------------------
    services_data = []
    service_user_company_pairs = set()
    today = date.today().isoformat()
    user_ids = list(user_id_map.values())

    for i in range(15):
        user_id = user_ids[i % len(user_ids)]
        company_id = company_ids[i % len(company_ids)]
        device_id = device_pool_by_company[company_id][i % len(device_pool_by_company[company_id])]
        service_user_company_pairs.add((user_id, company_id))
        services_data.append({
            "users_id": user_id,
            "task_id": 1,
            "ml_method_id": 2 if i < 7 else 1,
            "device_id": device_id,
            "company_id": company_id,
            "start_date": "2022-01-01" if i < 3 else "2023-01-01" if i < 7 else "2024-01-01",
            "end_date": "2023-12-31" if i < 5 else today if i < 7 else "2025-12-31"
        })

    marie_id = user_id_map["curie"]
    services_data.extend([
        {
            "users_id": marie_id,
            "task_id": 1,
            "ml_method_id": 1,
            "device_id": device_pool_by_company[company_map["Greyhound Dynamics"]][0],
            "company_id": company_map["Greyhound Dynamics"],
            "start_date": "2024-01-01",
            "end_date": "2025-12-31"
        },
        {
            "users_id": marie_id,
            "task_id": 1,
            "ml_method_id": 1,
            "device_id": device_pool_by_company[company_map["Poodle Analytics"]][0],
            "company_id": company_map["Poodle Analytics"],
            "start_date": "2024-01-01",
            "end_date": "2025-12-31"
        }
    ])
    service_user_company_pairs.update({
        (marie_id, company_map["Greyhound Dynamics"]),
        (marie_id, company_map["Poodle Analytics"])
    })

    inserted_services = supabase.table("service").insert(services_data).execute().data

    # ------------------------------
    # Insert Users_Ext
    # ------------------------------
    roles = supabase.table("role").select("id", "name").execute().data
    role_id_map = {r["name"]: r["id"] for r in roles}

    users_ext_data = []
    admin_user_id = 1
    for user_id, company_id in service_user_company_pairs:
        if user_id == admin_user_id:
            users_ext_data.extend([
                {"user_id": admin_user_id, "role_id": role_id_map["Super Admin"], "company_id": rafai_id, "is_active": True, "notes": "Admin test"},
                {"user_id": admin_user_id, "role_id": role_id_map["Owner"], "company_id": rafai_id, "is_active": True, "notes": "Admin test"},
                {"user_id": admin_user_id, "role_id": role_id_map["Admin"], "company_id": rafai_id, "is_active": True, "notes": "Admin test"},
                {"user_id": admin_user_id, "role_id": role_id_map["Client"], "company_id": rafai_id, "is_active": True, "notes": "Admin test"}
            ])
        elif any(user_id == c["owner_id"] for c in inserted_companies if c["id"] == company_id):
            users_ext_data.append({"user_id": user_id, "role_id": role_id_map["Owner"], "company_id": company_id, "is_active": True, "notes": None})
        else:
            users_ext_data.append({"user_id": user_id, "role_id": role_id_map["Client"], "company_id": company_id, "is_active": True, "notes": None})
    supabase.table("users_ext").insert(users_ext_data).execute()

    # ------------------------------
    # Insert Results
    # ------------------------------
    results_data = []
    mode_cycle = [3, 3, 1, 2]
    for i, service in enumerate(inserted_services):
        count = 4 if i < 2 else 2
        for j in range(count):
            mode_id = mode_cycle[j % len(mode_cycle)]
            results_data.append({
                "device_id": service["device_id"],
                "user_id": service["users_id"],
                "service_id": service["id"],
                "mode_id": mode_id,
                "is_dev": mode_id in [1, 2],
                "result_json": {"status": "ok", "value": j + 1}
            })
    supabase.table("results").insert(results_data).execute()

    # ------------------------------
    # Save Snapshot
    # ------------------------------
    all_data = {
        "users": inserted_users,
        "companies": inserted_companies,
        "services": inserted_services,
        "results": results_data,
        "devices": inserted_devices
    }
    output_dir = os.path.join(os.path.dirname(__file__), "DummyData")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"dummy_data_{date.today().isoformat()}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(all_data, f, indent=4)

    print(f"✅ Supabase reset and dummy data saved to {filepath}")

if __name__ == "__main__":
    reset_supabase_db()
