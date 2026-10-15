from sandbox import run_code

memories = [
    {"id": "m1", "content": "We use PostgreSQL", "source_type": "dialogue", "timestamp": "2026-10-01"},
    {"id": "m2", "content": "def add(a,b): return a+b", "source_type": "code", "timestamp": "2026-10-02"},
    {"id": "m3", "content": "Switched database to MongoDB", "source_type": "dialogue", "timestamp": "2026-10-05"},
]

print(run_code("from collections import Counter\nprint(Counter(m['source_type'] for m in memories))", memories=memories))
print(run_code("db=[m for m in memories if 'database' in m['content'].lower()]\nprint(max(db,key=lambda m:m['timestamp'])['content'])", memories=memories))
print(run_code("print(1/0)", memories=memories))