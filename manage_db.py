# neo4j_neighbors_example.py
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import os
from create_user_vector import (
    create_user_vector,
    create_user_vector_with_weights,
    parameter_weights,
    cosine_distance,
)

PARAMETERS = ['rooms', 'roommates', 'budget', 'months']

def get_driver(uri=None, user=None, password=None):
    uri = uri or os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')  # prefer IPv4 localhost
    user = user or os.getenv('NEO4J_USER', 'neo4j')
    password = password or os.getenv('NEO4J_PASSWORD', 'password')
    return GraphDatabase.driver(uri, auth=(user, password))

def ensure_constraints_and_index(session, dims):
    # Ensure a unique id constraint for Users
    session.run("""
        CREATE CONSTRAINT user_id_unique IF NOT EXISTS
        FOR (u:User) REQUIRE u.id IS UNIQUE
    """)
    # Create vector index for User.embedding if it doesn't exist
    result = session.run("""
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties
        WHERE type = 'VECTOR' AND entityType = 'NODE'
          AND 'User' IN labelsOrTypes AND 'embedding' IN properties
        RETURN name LIMIT 1
    """)
    if result.peek() is None:
        session.run("""
            CALL db.index.vector.createNodeIndex($name, 'User', 'embedding', $dims, 'cosine')
        """, name='user_vec_index', dims=dims)

def clear_users(session):
    session.run("MATCH (u:User) DETACH DELETE u")

def upsert_users(session, users, caps=None, use_weights=False, weights=None):
    weights = weights or parameter_weights
    rows = []
    for u in users:
        vec = create_user_vector_with_weights(u, PARAMETERS, weights, caps) if use_weights \
              else create_user_vector(u, PARAMETERS, caps)
        rows.append({
            'id': u['id'],
            'name': u.get('name'),
            'rooms': u.get('rooms'),
            'roommates': u.get('roommates'),
            'budget': u.get('budget'),
            'months': u.get('months'),
            'embedding': vec,
        })
    session.run("""
        UNWIND $rows AS row
        MERGE (u:User {id: row.id})
        SET u.name = row.name,
            u.rooms = row.rooms,
            u.roommates = row.roommates,
            u.budget = row.budget,
            u.months = row.months,
            u.embedding = row.embedding
    """, rows=rows)

def find_similar(session, vector, top_k=5, exclude_id=None):
    records = session.run("""
        CALL db.index.vector.queryNodes($indexName, $k, $vector)
        YIELD node, score
        WHERE $excludeId IS NULL OR node.id <> $excludeId
        RETURN node.id AS id, node.name AS name, score
        ORDER BY score DESC
        LIMIT $k
    """, indexName='user_vec_index', k=top_k, vector=vector, excludeId=exclude_id)
    return [r.data() for r in records]

def find_similar_local(users, query_user, caps=None, use_weights=False, weights=None, top_k=5):
    weights = weights or parameter_weights
    if use_weights:
        qvec = create_user_vector_with_weights(query_user, PARAMETERS, weights, caps)
    else:
        qvec = create_user_vector(query_user, PARAMETERS, caps)
    results = []
    for u in users:
        if u['id'] == query_user.get('id'):
            continue
        if use_weights:
            uvec = create_user_vector_with_weights(u, PARAMETERS, weights, caps)
        else:
            uvec = create_user_vector(u, PARAMETERS, caps)
        # cosine_distance returns distance; convert to similarity
        sim = 1.0 - cosine_distance(qvec, uvec)
        results.append({'id': u['id'], 'name': u.get('name'), 'score': sim})
    results.sort(key=lambda r: r['score'], reverse=True)
    return results[:top_k]

def sample_users():
    return [
        {'id': 'u1', 'name': 'Alice', 'rooms': 1, 'roommates': 1, 'budget': 10000, 'months': 12},
        {'id': 'u2', 'name': 'Bob',   'rooms': 2, 'roommates': 2, 'budget': 12000, 'months': 6},
        {'id': 'u3', 'name': 'Rich guy',  'rooms': 4, 'roommates': 2, 'budget':  60000, 'months': 12},
        {'id': 'u4', 'name': 'Dave',  'rooms': 2, 'roommates': 2, 'budget': 15000, 'months':  9},
        {'id': 'u5', 'name': 'Eve',   'rooms': 1, 'roommates': 0, 'budget': 11000, 'months': 24},
    ]

def build_test_db():
    uri = os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'password')  # change for your instance
    caps = {'budget': 200000, 'months': 36}  # adjust if you update normalization caps

    users = sample_users()

    try:
        with get_driver(uri, user, password) as driver:
            with driver.session() as session:
                ensure_constraints_and_index(session, dims=len(PARAMETERS))
                clear_users(session)  # optional clean start for the demo
                upsert_users(session, users, caps=caps, use_weights=False)

                # Query: find neighbors for 'u1'
                query_user = users[0]
                query_vec = create_user_vector(query_user, PARAMETERS, caps)
                results = find_similar(session, query_vec, top_k=3, exclude_id=query_user['id'])
                return results
    except ServiceUnavailable as e:
        print("Neo4j connection failed. Falling back to local similarity computation.")
        print(f"Reason: {e}")
        query_user = sample_users()[0]
        results = find_similar_local(sample_users(), query_user, caps=caps, use_weights=False, top_k=3)
        return results

if __name__ == '__main__':
    results = build_test_db()
    for r in results:
        print(f"{r['id']} {r['name']} -> similarity={r['score']:.4f}")
