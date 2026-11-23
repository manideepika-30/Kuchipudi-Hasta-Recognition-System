import urllib.request
import json

req = urllib.request.Request("http://127.0.0.1:8000/api/curriculum/modules")
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))
        print(f"Total modules returned: {len(data)}")
        for m in data:
            print(f"Module {m['module_number']}: {m['title_en']} ({m['title_te']}) | Lessons: {len(m['lessons'])}")
            if m['lessons']:
                first_les = m['lessons'][0]
                print(f"   First lesson: {first_les['name_en']} ({first_les['name_te']}) - Image: {first_les['image_url']}")
except Exception as e:
    print(f"Error calling API: {e}")
