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
from manage_logging import (
    setup_logger,
    log_neo4j_query,
    log_vector_operation,
    log_similarity_results,
    log_database_stats
)
import dotenv
dotenv.load_dotenv()

# Setup logger
logger = setup_logger("roommate_db", "INFO")

PARAMETERS = ['rooms', 'roommates', 'budget', 'months']

def get_driver(uri=None, user=None, password=None):
    uri = uri or os.getenv('NEO4J_URI')  # prefer IPv4 localhost
    user = user or os.getenv('NEO4J_USERNAME')
    password = password or os.getenv('NEO4J_PASSWORD')
    return GraphDatabase.driver(uri, auth=(user, password))

def ensure_constraints_and_index(session, dims):
    """Ensure required constraints and vector index exist for User nodes."""
    logger.debug(f"Setting up database constraints and indexes for {dims}-dimensional vectors")
    
    try:
        # Ensure a unique id constraint for Users (proper Neo4j syntax)
        constraint_query = """
            CREATE CONSTRAINT user_id_unique IF NOT EXISTS
            FOR (u:User) REQUIRE u.id IS UNIQUE
        """
        log_neo4j_query(logger, constraint_query)
        session.run(constraint_query)
        logger.info("✓ User ID uniqueness constraint ensured")
        
        # Check if vector index exists
        index_check_query = """
            SHOW INDEXES
            YIELD name, type, entityType, labelsOrTypes, properties
            WHERE type = 'VECTOR' AND entityType = 'NODE'
              AND 'User' IN labelsOrTypes AND 'embedding' IN properties
            RETURN name LIMIT 1
        """
        log_neo4j_query(logger, index_check_query)
        result = session.run(index_check_query)
        
        if result.peek() is None:
            # Create vector index for User.embedding using new syntax
            index_create_query = """
                CREATE VECTOR INDEX user_vec_index IF NOT EXISTS
                FOR (u:User) ON (u.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                }}
            """
            log_neo4j_query(logger, index_create_query, {"dims": dims})
            session.run(index_create_query, dims=dims)
            logger.info(f"✓ Vector index 'user_vec_index' created with {dims} dimensions")
        else:
            logger.info("✓ Vector index already exists")
            logger.debug("Skipping vector index creation - index already present")
            
    except Neo4jError as e:
        logger.warning(f"Could not ensure constraints/indexes: {e}")
        logger.debug("Continuing execution - constraints might already exist")
        # Continue execution - constraints might already exist

def clear_users(session):
    """Remove all User nodes and their relationships."""
    clear_query = "MATCH (u:User) DETACH DELETE u"
    log_neo4j_query(logger, clear_query)
    
    result = session.run(clear_query)
    logger.info("✓ All existing users cleared from database")
    logger.debug("Database cleanup completed - all User nodes and relationships removed")

def upsert_users(session, users, caps=None, use_weights=False, weights=None):
    """Insert or update users with their preference vectors."""
    weights = weights or parameter_weights
    rows = []
    
    logger.debug(f"Processing {len(users)} users for database upsert")
    logger.debug(f"Using weights: {use_weights}, Vector caps: {caps}")
    
    for u in users:
        vec = create_user_vector_with_weights(u, PARAMETERS, weights, caps) if use_weights \
              else create_user_vector(u, PARAMETERS, caps)
        
        log_vector_operation(logger, "Created user vector", len(vec), u['id'])
        
        rows.append({
            'id': u['id'],
            'name': u.get('name'),
            'rooms': u.get('rooms'),
            'roommates': u.get('roommates'),
            'budget': u.get('budget'),
            'months': u.get('months'),
            'embedding': vec,
        })
    
    upsert_query = """
        UNWIND $rows AS row
        MERGE (u:User {id: row.id})
        SET u.name = row.name,
            u.rooms = row.rooms,
            u.roommates = row.roommates,
            u.budget = row.budget,
            u.months = row.months,
            u.embedding = row.embedding
        RETURN count(u) as created
    """
    
    log_neo4j_query(logger, upsert_query, {"rows_count": len(rows)})
    result = session.run(upsert_query, rows=rows)
    
    count = result.single()['created']
    logger.info(f"✓ {count} users upserted successfully")
    
    log_database_stats(logger, {"nodes_created": count, "vectors_generated": len(rows)})

def find_similar(session, vector, top_k=5, exclude_id=None):
    """Find similar users using vector similarity search."""
    log_vector_operation(logger, "Executing similarity search", len(vector), exclude_id)
    
    similarity_query = """
        CALL db.index.vector.queryNodes($indexName, $k, $vector)
        YIELD node, score
        WHERE $excludeId IS NULL OR node.id <> $excludeId
        RETURN node.id AS id, node.name AS name, score
        ORDER BY score DESC
        LIMIT $k
    """
    
    params = {
        'indexName': 'user_vec_index', 
        'k': top_k, 
        'vector': vector, 
        'excludeId': exclude_id
    }
    log_neo4j_query(logger, similarity_query, {**params, 'vector': f'[{len(vector)} elements]'})
    
    records = session.run(similarity_query, **params)
    results = [r.data() for r in records]
    
    if exclude_id:
        log_similarity_results(logger, exclude_id, results, top_k)
    
    return results

def find_similar_local(users, query_user, caps=None, use_weights=False, weights=None, top_k=5):
    """Find similar users using local computation (fallback method)."""
    weights = weights or parameter_weights
    query_id = query_user.get('id')
    
    logger.debug(f"Computing local similarity for user {query_id} against {len(users)} users")
    logger.debug(f"Using weights: {use_weights}, caps: {caps}")
    
    if use_weights:
        qvec = create_user_vector_with_weights(query_user, PARAMETERS, weights, caps)
    else:
        qvec = create_user_vector(query_user, PARAMETERS, caps)
    
    log_vector_operation(logger, "Generated query vector", len(qvec), query_id)
    
    results = []
    for u in users:
        if u['id'] == query_id:
            continue
            
        if use_weights:
            uvec = create_user_vector_with_weights(u, PARAMETERS, weights, caps)
        else:
            uvec = create_user_vector(u, PARAMETERS, caps)
        
        # cosine_distance returns distance; convert to similarity
        distance = cosine_distance(qvec, uvec)
        sim = 1.0 - distance
        
        logger.debug(f"Similarity between {query_id} and {u['id']}: {sim:.4f} (distance: {distance:.4f})")
        results.append({'id': u['id'], 'name': u.get('name'), 'score': sim})
    
    results.sort(key=lambda r: r['score'], reverse=True)
    final_results = results[:top_k]
    
    log_similarity_results(logger, query_id, final_results, top_k)
    return final_results

def clean_db():
    """Clean the entire database after user confirmation."""
    logger.warning("Database cleaning requested - this will delete ALL data")
    confirmation = input("⚠️  This will delete ALL data in the database. Are you sure? (type 'YES' to confirm): ")
    
    if confirmation != 'YES':
        logger.info("Database cleaning cancelled by user")
        return False
        
    logger.info("Starting complete database cleanup...")
    
    try:
        with get_driver() as driver:
            with driver.session() as session:
                # Delete all nodes and relationships
                delete_query = "MATCH (n) DETACH DELETE n"
                log_neo4j_query(logger, delete_query)
                session.run(delete_query)
                logger.info("✓ All nodes and relationships deleted")
                
                # Drop all indexes (except built-in ones)
                indexes_query = """
                    SHOW INDEXES
                    YIELD name, type
                    WHERE type = 'VECTOR'
                    RETURN name
                """
                log_neo4j_query(logger, indexes_query)
                indexes_result = session.run(indexes_query)
                
                dropped_indexes = 0
                for record in indexes_result:
                    index_name = record['name']
                    try:
                        drop_index_query = f"DROP INDEX {index_name}"
                        log_neo4j_query(logger, drop_index_query)
                        session.run(drop_index_query)
                        logger.info(f"✓ Dropped index: {index_name}")
                        dropped_indexes += 1
                    except Neo4jError as e:
                        logger.debug(f"Could not drop index {index_name}: {e}")
                
                # Drop all constraints
                constraints_query = """
                    SHOW CONSTRAINTS
                    YIELD name
                    RETURN name
                """
                log_neo4j_query(logger, constraints_query)
                constraints_result = session.run(constraints_query)
                
                dropped_constraints = 0
                for record in constraints_result:
                    constraint_name = record['name']
                    try:
                        drop_constraint_query = f"DROP CONSTRAINT {constraint_name}"
                        log_neo4j_query(logger, drop_constraint_query)
                        session.run(drop_constraint_query)
                        logger.info(f"✓ Dropped constraint: {constraint_name}")
                        dropped_constraints += 1
                    except Neo4jError as e:
                        logger.debug(f"Could not drop constraint {constraint_name}: {e}")
                        
                logger.info("✅ Database cleaned successfully!")
                log_database_stats(logger, {
                    "indexes_dropped": dropped_indexes,
                    "constraints_dropped": dropped_constraints
                })
                return True
                
    except Exception as e:
        logger.error(f"Error cleaning database: {e}")
        return False


def sample_users():
    """Generate a diverse set of test users for recommendation testing."""
    users = [
        {'id': 'u1', 'name': 'Alice (Budget Student)', 'rooms': 1, 'roommates': 1, 'budget': 8000, 'months': 12},
        {'id': 'u2', 'name': 'Bob (Shared Apartment)', 'rooms': 2, 'roommates': 2, 'budget': 12000, 'months': 6},
        {'id': 'u3', 'name': 'Charlie (Luxury Seeker)', 'rooms': 4, 'roommates': 2, 'budget': 60000, 'months': 12},
        {'id': 'u4', 'name': 'Dave (Professional)', 'rooms': 2, 'roommates': 1, 'budget': 15000, 'months': 9},
        {'id': 'u5', 'name': 'Eve (Solo Living)', 'rooms': 1, 'roommates': 0, 'budget': 11000, 'months': 24},
        {'id': 'u6', 'name': 'Frank (Budget Conscious)', 'rooms': 1, 'roommates': 1, 'budget': 9000, 'months': 12},
        {'id': 'u7', 'name': 'Grace (Short Term)', 'rooms': 2, 'roommates': 1, 'budget': 14000, 'months': 3},
        {'id': 'u8', 'name': 'Henry (Family Space)', 'rooms': 3, 'roommates': 3, 'budget': 25000, 'months': 18},
    ]
    
    logger.debug(f"Generated {len(users)} test users with diverse preferences")
    for user in users:
        logger.debug(f"User {user['id']}: {user['name']} - {user['rooms']}R, {user['roommates']}RM, ₽{user['budget']}, {user['months']}M")
    
    return users

def build_test_db_and_find_recommendations(
    top_k=3,
    use_weights=False,
    caps=None,
    weights=None,
    clear_db_first=True,
    test_users=None,
    verbose=True
):
    """
    Build test database and find top-k roommate recommendations for each user.
    
    Args:
        top_k (int): Number of recommendations per user. Use -1 for all available users.
        use_weights (bool): Whether to use weighted vectors for similarity calculation.
        caps (dict): Normalization caps for user properties. Default: {'budget': 200000, 'months': 36}
        weights (dict): Weights for vector components. Default: parameter_weights from create_user_vector.
        clear_db_first (bool): Whether to clear existing users before inserting test data.
        test_users (list): Custom list of test users. Default: sample_users()
        verbose (bool): Whether to log detailed information about each user's recommendations.
        
    Returns:
        dict: Dictionary mapping user IDs to their recommendation lists.
    """
    # Set defaults
    caps = caps or {'budget': 200000, 'months': 36}  # normalization caps
    weights = weights or parameter_weights
    users = test_users or sample_users()
    
    # Handle special case where top_k = -1 means "all users"
    effective_top_k = len(users) - 1 if top_k == -1 else top_k
    
    logger.info(f"🏠 Building test database with {len(users)} users")
    logger.info(f"Configuration: top_k={'all' if top_k == -1 else top_k}, "
               f"weighted={use_weights}, clear_first={clear_db_first}")
    logger.debug(f"Vector normalization caps: {caps}")
    logger.debug(f"Parameters used: {PARAMETERS}")
    logger.debug(f"Weights: {weights if use_weights else 'None (unweighted)'}")

    try:
        with get_driver() as driver:
            with driver.session() as session:
                # Setup database
                logger.info("Setting up database schema...")
                ensure_constraints_and_index(session, dims=len(PARAMETERS))
                
                if clear_db_first:
                    clear_users(session)
                else:
                    logger.info("Skipping database clear - keeping existing data")
                
                upsert_users(session, users, caps=caps, use_weights=use_weights, weights=weights)
                
                top_k_desc = "all available" if top_k == -1 else str(effective_top_k)
                logger.info(f"🔍 Finding {top_k_desc} roommate recommendations for each user")
                
                all_results = {}
                successful_queries = 0
                
                # Find recommendations for each user
                for user in users:
                    if verbose:
                        logger.info(f"👤 {user['name']} (ID: {user['id']})")
                        logger.info(f"   Preferences: {user['rooms']} rooms, {user['roommates']} roommates, "
                                  f"₽{user['budget']} budget, {user['months']} months")
                    else:
                        logger.debug(f"Processing user {user['id']}: {user['name']}")
                    
                    # Create query vector (use same weighting as database vectors)
                    if use_weights:
                        query_vec = create_user_vector_with_weights(user, PARAMETERS, weights, caps)
                    else:
                        query_vec = create_user_vector(user, PARAMETERS, caps)
                    
                    # Find similar users
                    results = find_similar(session, query_vec, top_k=effective_top_k, exclude_id=user['id'])
                    all_results[user['id']] = results
                    
                    if results:
                        if verbose:
                            logger.info("   💫 Top recommendations:")
                            for i, r in enumerate(results, 1):
                                similarity_pct = r['score'] * 100
                                logger.info(f"      {i}. {r['name']} (ID: {r['id']}) - {similarity_pct:.1f}% match")
                        successful_queries += 1
                    else:
                        if verbose:
                            logger.warning("   ❌ No recommendations found")
                        else:
                            logger.debug(f"No recommendations found for user {user['id']}")
                
                logger.debug(f"Recommendation generation completed: {successful_queries}/{len(users)} users processed successfully")
                return all_results
                
    except ServiceUnavailable as e:
        logger.error("❌ Neo4j connection failed. Falling back to local similarity computation.")
        logger.error(f"Connection error: {e}")
        logger.info("Switching to local computation mode...")
        
        top_k_desc = "all available" if top_k == -1 else str(effective_top_k)
        logger.info(f"🔍 Finding {top_k_desc} roommate recommendations (local computation)")
        
        all_results = {}
        successful_queries = 0
        
        for user in users:
            if verbose:
                logger.info(f"👤 {user['name']} (ID: {user['id']})")
                logger.info(f"   Preferences: {user['rooms']} rooms, {user['roommates']} roommates, "
                          f"₽{user['budget']} budget, {user['months']} months")
            else:
                logger.debug(f"Processing user {user['id']}: {user['name']}")
            
            results = find_similar_local(users, user, caps=caps, use_weights=use_weights, 
                                       weights=weights, top_k=effective_top_k)
            all_results[user['id']] = results
            
            if results:
                if verbose:
                    logger.info("   💫 Top recommendations:")
                    for i, r in enumerate(results, 1):
                        similarity_pct = r['score'] * 100
                        logger.info(f"      {i}. {r['name']} (ID: {r['id']}) - {similarity_pct:.1f}% match")
                successful_queries += 1
            else:
                if verbose:
                    logger.warning("   ❌ No recommendations found")
                else:
                    logger.debug(f"No recommendations found for user {user['id']}")
            
        logger.debug(f"Local computation completed: {successful_queries}/{len(users)} users processed successfully")
        return all_results
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
        return {}

if __name__ == '__main__':
    logger.info("🎯 Neo4j Roommate Recommendation System Test")
    logger.info("=" * 50)
    
    # Clean database before testing
    logger.info("Starting database cleanup...")
    clean_db()
    
    logger.info("\nStarting recommendation system test...")
    all_recommendations = build_test_db_and_find_recommendations(top_k=-1)
    
    logger.info("\n📊 Test Summary:")
    logger.info(f"Processed {len(all_recommendations)} users")
    
    total_recommendations = sum(len(recs) for recs in all_recommendations.values())
    logger.info(f"Generated {total_recommendations} total recommendations")
    
    if all_recommendations:
        avg_recommendations = total_recommendations / len(all_recommendations)
        logger.debug(f"Average recommendations per user: {avg_recommendations:.1f}")
        
        # Log some statistics
        recommendation_counts = [len(recs) for recs in all_recommendations.values()]
        max_recs = max(recommendation_counts) if recommendation_counts else 0
        min_recs = min(recommendation_counts) if recommendation_counts else 0
        logger.debug(f"Recommendation range: {min_recs}-{max_recs} per user")
    
    logger.info("\n✅ Test completed successfully!")
    logger.debug("System test finished - all operations completed")
