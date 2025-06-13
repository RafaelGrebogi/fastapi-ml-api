import json
import sys
import os
from datetime import date

# Add the root folder to sys.path so you can import supabase_client
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from supabase_client import supabase


# SUPABASE_URL = "YOUR_SUPABASE_URL"
# SUPABASE_KEY = "YOUR_SUPABASE_KEY"

# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def reset_supabase_db():
    # ------------------------------
    # Clear Tables (except static and users.id=1)
    # ------------------------------
    supabase.table("results").delete().neq("id", -1).execute()
    supabase.table("service").delete().neq("id", -1).execute()
    supabase.table("company").delete().neq("id", -1).execute()
    supabase.table("users").delete().neq("id", 1).execute()

    # ------------------------------
    # Insert Users (Albert Einstein is admin)
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
    ]
    inserted_users = supabase.table("users").insert(users_data).execute().data

    # Map usernames to real IDs
    user_id_map = {user["username"]: user["id"] for user in inserted_users}

    # ------------------------------
    # Insert Companies
    # ------------------------------
    companies_data = [
        {"name": "Greyhound Dynamics", "owner_id": user_id_map["curie"], "email": "greyhound1@corp.com", "phone": "+61211111111", "address": "1 Street NSW", "is_active": True},
        {"name": "BorderCollie Systems", "owner_id": user_id_map["newton"], "email": "bcollie@corp.com", "phone": "+61211111112", "address": "2 Street NSW", "is_active": True},
        {"name": "Bulldog AI", "owner_id": user_id_map["bohr"], "email": "bulldog@corp.com", "phone": "+61211111113", "address": "3 Street NSW", "is_active": True},
        {"name": "Poodle Analytics", "owner_id": user_id_map["galileo"], "email": "poodle@corp.com", "phone": "+61211111114", "address": "4 Street NSW", "is_active": True},
        {"name": "BeagleTech", "owner_id": user_id_map["feynman"], "email": "beagle@corp.com", "phone": "+61211111115", "address": "5 Street NSW", "is_active": False},
    ]
    inserted_companies = supabase.table("company").insert(companies_data).execute().data
    company_map = {c["name"]: c["id"] for c in inserted_companies}
    company_ids = [c["id"] for c in inserted_companies]

    # ------------------------------
    # Insert Services
    # ------------------------------
    services_data = []
    today = date.today().isoformat()
    for i in range(15):
        start_date = "2022-01-01" if i < 3 else "2023-01-01" if i < 7 else "2024-01-01"
        end_date = "2023-12-31" if i < 5 else today if i < 7 else "2025-12-31"
        services_data.append({
            "users_id": list(user_id_map.values())[i % len(user_id_map)],
            "task_id": 1,
            "ml_method_id": 2 if i < 7 else 1,
            "device_id": 1 if i < 7 else 2,
            "company_id": company_ids[i % len(company_ids)],
            "start_date": start_date,
            "end_date": end_date
        })

    # Add two specific services for Marie Curie at Greyhound and Poodle
    marie_id = user_id_map["curie"]
    services_data.append({
        "users_id": marie_id,
        "task_id": 1,
        "ml_method_id": 1,
        "device_id": 1,
        "company_id": company_map["Greyhound Dynamics"],
        "start_date": "2024-01-01",
        "end_date": "2025-12-31"
    })
    services_data.append({
        "users_id": marie_id,
        "task_id": 1,
        "ml_method_id": 1,
        "device_id": 2,
        "company_id": company_map["Poodle Analytics"],
        "start_date": "2024-01-01",
        "end_date": "2025-12-31"
    })

    inserted_services = supabase.table("service").insert(services_data).execute().data

    # ------------------------------
    # Insert Results (favoring mode_id = 3)
    # ------------------------------
    results_data = []
    result_id = 1
    mode_cycle = [3, 1, 2, 3]  # prioritise production mode

    for service in inserted_services:
        count = 4 if service["id"] in [inserted_services[0]["id"], inserted_services[1]["id"]] else 2
        for i in range(count):
            mode_id = mode_cycle[i % len(mode_cycle)]
            results_data.append({
                "device_id": service["device_id"],
                "user_id": service["users_id"],
                "service_id": service["id"],
                "mode_id": mode_id,
                "is_dev": mode_id in [1, 2],
                "result_json": {"status": "ok", "value": i + 1}
            })
            result_id += 1

    supabase.table("results").insert(results_data).execute()

    # ------------------------------
    # Save Final Inserted Data
    # ------------------------------
    all_data = {
        "users": inserted_users,
        "companies": inserted_companies,
        "services": inserted_services,
        "results": results_data
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
