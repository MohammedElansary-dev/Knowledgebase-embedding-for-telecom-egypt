import requests

print(
    requests.post(
        "http://0.0.0.0:10000",
        json={
            "message": "Start typing here"
        }
    ).json()
)
