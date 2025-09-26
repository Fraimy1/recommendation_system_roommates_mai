# neo4j_neighbors_example.py
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import os
from create_user_vector import (
    create_user_vector,
    create_group_vector_with_weights,
    group_parameter_weights,
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
    """Ensure required constraints and vector indexes exist for User and Group nodes."""
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
        
        # Ensure a unique constraint for Parameter nodes per (userId, name)
        param_constraint_query = """
            CREATE CONSTRAINT parameter_unique IF NOT EXISTS
            FOR (p:Parameter) REQUIRE (p.userId, p.name) IS UNIQUE
        """
        log_neo4j_query(logger, param_constraint_query)
        session.run(param_constraint_query)
        logger.info("✓ Parameter uniqueness constraint ensured (userId, name)")

        # Ensure a unique id constraint for Groups
        group_constraint_query = """
            CREATE CONSTRAINT group_id_unique IF NOT EXISTS
            FOR (g:Group) REQUIRE g.id IS UNIQUE
        """
        log_neo4j_query(logger, group_constraint_query)
        session.run(group_constraint_query)
        logger.info("✓ Group ID uniqueness constraint ensured")

        # Ensure a unique constraint for GroupParameter (groupId, name)
        gparam_constraint_query = """
            CREATE CONSTRAINT group_parameter_unique IF NOT EXISTS
            FOR (p:GroupParameter) REQUIRE (p.groupId, p.name) IS UNIQUE
        """
        log_neo4j_query(logger, gparam_constraint_query)
        session.run(gparam_constraint_query)
        logger.info("✓ GroupParameter uniqueness constraint ensured (groupId, name)")
        
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

        # Check if GROUP vector index exists
        gindex_check_query = """
            SHOW INDEXES
            YIELD name, type, entityType, labelsOrTypes, properties
            WHERE type = 'VECTOR' AND entityType = 'NODE'
              AND 'Group' IN labelsOrTypes AND 'embedding' IN properties
            RETURN name LIMIT 1
        """
        log_neo4j_query(logger, gindex_check_query)
        gresult = session.run(gindex_check_query)

        if gresult.peek() is None:
            gindex_create_query = """
                CREATE VECTOR INDEX group_vec_index IF NOT EXISTS
                FOR (g:Group) ON (g.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                }}
            """
            log_neo4j_query(logger, gindex_create_query, {"dims": dims})
            session.run(gindex_create_query, dims=dims)
            logger.info(f"✓ Vector index 'group_vec_index' created with {dims} dimensions")
        else:
            logger.info("✓ Group vector index already exists")
            logger.debug("Skipping group vector index creation - index already present")
            
    except Neo4jError as e:
        logger.warning(f"Could not ensure constraints/indexes: {e}")
        logger.debug("Continuing execution - constraints might already exist")
        # Continue execution - constraints might already exist

def clear_users(session):
    """Remove all User, Group nodes and their Parameter nodes and relationships."""
    # Delete Parameter nodes attached to users first to avoid orphans
    clear_params_query = "MATCH (:User)-[:HAS_PARAMETER]->(p:Parameter) DETACH DELETE p"
    log_neo4j_query(logger, clear_params_query)
    session.run(clear_params_query)

    # Delete GroupParameter nodes
    clear_gparams_query = "MATCH (:Group)-[:HAS_PARAMETER]->(p:GroupParameter) DETACH DELETE p"
    log_neo4j_query(logger, clear_gparams_query)
    session.run(clear_gparams_query)

    # Now delete groups
    clear_groups_query = "MATCH (g:Group) DETACH DELETE g"
    log_neo4j_query(logger, clear_groups_query)
    session.run(clear_groups_query)

    # Now delete users
    clear_users_query = "MATCH (u:User) DETACH DELETE u"
    log_neo4j_query(logger, clear_users_query)
    session.run(clear_users_query)

    logger.info("✓ All existing users, groups, and their parameters cleared from database")
    logger.debug("Database cleanup completed - all User, Group and Parameter nodes removed")

def upsert_users(session, users, caps=None, use_weights=False, weights=None):
    """Insert or update users along with their single-member groups and parameters."""
    weights = weights or group_parameter_weights
    rows = []
    
    logger.debug(f"Processing {len(users)} users for database upsert")
    logger.debug(f"Using weights: {use_weights}, Vector caps: {caps}")
    
    for u in users:
        # Single-member group id and name
        group_id = f"g_{u['id']}"
        group_name = f"Group of {u.get('name') or u['id']}"

        # Prepare parameter list as separate nodes (user and group)
        param_list = [{
            'name': p,
            'value': u.get(p)
        } for p in PARAMETERS]

        group_values = {p: u.get(p) for p in PARAMETERS}
        gvec = create_group_vector_with_weights(group_values, PARAMETERS, weights, caps) if use_weights \
              else create_user_vector(group_values, PARAMETERS, caps)

        log_vector_operation(logger, "Created group vector", len(gvec), group_id)

        rows.append({
            'user': {
                'id': u['id'],
                'name': u.get('name'),
                'parameters': param_list,
            },
            'group': {
                'id': group_id,
                'name': group_name,
                'embedding': gvec,
                'parameters': param_list,
            }
        })
    
    upsert_query = """
        UNWIND $rows AS row
        MERGE (u:User {id: row.user.id})
        SET u.name = row.user.name
        WITH u, row
        UNWIND row.user.parameters AS param
        MERGE (p:Parameter {userId: row.user.id, name: param.name})
        SET p.value = param.value
        MERGE (u)-[:HAS_PARAMETER]->(p)
        WITH u, row
        MERGE (g:Group {id: row.group.id})
        SET g.name = row.group.name,
            g.embedding = row.group.embedding
        MERGE (u)-[:MEMBER_OF]->(g)
        WITH g, row
        UNWIND row.group.parameters AS gparam
        MERGE (gp:GroupParameter {groupId: row.group.id, name: gparam.name})
        SET gp.value = gparam.value
        MERGE (g)-[:HAS_PARAMETER]->(gp)
        WITH DISTINCT g
        RETURN count(g) as created
    """
    
    log_neo4j_query(logger, upsert_query, {"rows_count": len(rows)})
    result = session.run(upsert_query, rows=rows)
    
    count = result.single()['created']
    logger.info(f"✓ {count} groups upserted successfully (and linked to users)")
    
    log_database_stats(logger, {"groups_created": count, "group_vectors_generated": len(rows)})

def find_similar(session, vector, top_k=5, exclude_id=None):
    """Find similar groups using vector similarity search."""
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
        'indexName': 'group_vec_index', 
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
    """Find similar groups using local computation (fallback method)."""
    weights = weights or group_parameter_weights
    query_user_id = query_user.get('id')
    query_group_id = f"g_{query_user_id}"
    
    logger.debug(f"Computing local similarity for group {query_group_id} against {len(users)} groups")
    logger.debug(f"Using weights: {use_weights}, caps: {caps}")
    
    group_values = {p: query_user.get(p) for p in PARAMETERS}
    if use_weights:
        qvec = create_group_vector_with_weights(group_values, PARAMETERS, weights, caps)
    else:
        qvec = create_user_vector(group_values, PARAMETERS, caps)
    
    log_vector_operation(logger, "Generated group query vector", len(qvec), query_group_id)
    
    results = []
    for u in users:
        uid = u['id']
        gid = f"g_{uid}"
        if gid == query_group_id:
            continue
            
        values = {p: u.get(p) for p in PARAMETERS}
        if use_weights:
            uvec = create_group_vector_with_weights(values, PARAMETERS, weights, caps)
        else:
            uvec = create_user_vector(values, PARAMETERS, caps)
        
        # cosine_distance returns distance; convert to similarity
        distance = cosine_distance(qvec, uvec)
        sim = 1.0 - distance
        
        logger.debug(f"Similarity between {query_group_id} and {gid}: {sim:.4f} (distance: {distance:.4f})")
        results.append({'id': gid, 'name': f"Group of {u.get('name') or uid}", 'score': sim})
    
    results.sort(key=lambda r: r['score'], reverse=True)
    final_results = results[:top_k]
    
    log_similarity_results(logger, query_group_id, final_results, top_k)
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
        use_weights (bool): Whether to use group weighted vectors for similarity calculation.
        caps (dict): Normalization caps for user properties. Default: {'budget': 200000, 'months': 36}
        weights (dict): Group weights for vector components. Default: group_parameter_weights.
        clear_db_first (bool): Whether to clear existing users before inserting test data.
        test_users (list): Custom list of test users. Default: sample_users()
        verbose (bool): Whether to log detailed information about each user's recommendations.
        
    Returns:
        dict: Dictionary mapping user IDs to their recommendation lists.
    """
    # Set defaults
    caps = caps or {'budget': 200000, 'months': 36}  # normalization caps
    weights = weights or group_parameter_weights
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
                
                # Find recommendations for each user's GROUP
                for user in users:
                    if verbose:
                        logger.info(f"👤 {user['name']} (ID: {user['id']})")
                        logger.info(f"   Preferences: {user['rooms']} rooms, {user['roommates']} roommates, "
                                  f"₽{user['budget']} budget, {user['months']} months")
                    else:
                        logger.debug(f"Processing user {user['id']}: {user['name']}")
                    
                    # Create group query vector (consistent with database vectors)
                    group_values = {p: user.get(p) for p in PARAMETERS}
                    if use_weights:
                        query_vec = create_group_vector_with_weights(group_values, PARAMETERS, weights, caps)
                    else:
                        query_vec = create_user_vector(group_values, PARAMETERS, caps)
                    
                    # Find similar groups (exclude this user's group)
                    group_id = f"g_{user['id']}"
                    results = find_similar(session, query_vec, top_k=effective_top_k, exclude_id=group_id)
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
