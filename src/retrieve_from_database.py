from src.enums.database_type_enum import DatabaseType
from src.enums.transformer_library_enum import TransformerLibrary
from src.models.vector_database_info import VectorDatabaseInfo

def get_search_type(*, search_method: str):
    # Sprawdzenie w DatabaseType
    for db_type in DatabaseType:
        if search_method == db_type.display_name:  # lub db_type.name, jeśli chcesz użyć nazwy enuma
            return db_type

    # Sprawdzenie w TransformerLibrary
    for library_type in TransformerLibrary:
        if search_method == library_type.display_name:
            return library_type

    # Jeśli nie znaleziono
    return None

def retrieve_from_database(*, vector_database_instance: VectorDatabaseInfo, search_method: str, query: str):
    search_type = get_search_type(search_method=search_method)

    if search_type is DatabaseType:
        vector_database_instance.retrieve_from_database(query=query)
    elif search_type is TransformerLibrary:
        return None

    return None