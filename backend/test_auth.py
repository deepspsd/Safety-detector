"""
Run this while uvicorn is running to find the exact auth error.
Usage: venv\Scripts\python test_auth.py
"""
import urllib.request, urllib.error, urllib.parse, json, sys

BASE = "http://localhost:8000"

def req(method, path, body=None, headers={}):
    url = BASE + path
    data = body.encode() if isinstance(body, str) else body
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

print("=" * 55)
print("OccuSafe Auth Diagnostics")
print("=" * 55)

# 1. Health
s, b = req("GET", "/health")
print(f"\n[1] GET /health → {s}: {b[:80]}")

# 2. Try signup
payload = json.dumps({"email":"diag@test.com","password":"test1234","name":"Diag User","role":"Construction Worker"})
s, b = req("POST", "/auth/signup", payload, {"Content-Type":"application/json"})
print(f"\n[2] POST /auth/signup → {s}")
print(f"    Body: {b[:300]}")

# 3. Try login with form-data (OAuth2 form)
form = urllib.parse.urlencode({"username":"diag@test.com","password":"test1234"})
s, b = req("POST", "/auth/login", form, {"Content-Type":"application/x-www-form-urlencoded"})
print(f"\n[3] POST /auth/login → {s}")
print(f"    Body: {b[:300]}")

# 4. Try /auth/me without token
s, b = req("GET", "/auth/me")
print(f"\n[4] GET /auth/me (no token) → {s}: {b[:80]}")

print("\n" + "=" * 55)
if s == 0:
    print("BACKEND IS DOWN — uvicorn is not running or crashed.")
    print("Check the uvicorn terminal for error messages.")
else:
    print("Backend is responding. Check errors above.")
print("=" * 55)
