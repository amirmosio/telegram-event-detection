from peewee import SqliteDatabase, OperationalError, DoesNotExist
from .models import User, UserCategorieMap, UserGroupMap


class QueryManager:
    def __init__(self) -> None:
        self.db = SqliteDatabase('peewee_database.db')
        self.db.connect()

        
        try:
            self.db.create_tables([User])
        except OperationalError:
            print("Table already exists!")

        try:
            self.db.create_tables([UserCategorieMap])
        except OperationalError:
            print("Table already exists!")

        try:
            self.db.create_tables([UserGroupMap])
        except OperationalError:
            print("Table already exists!")


    def check_if_user_has_been_registered(self, telegram_id):
        try:
            _ = User.get(User.telegram_id == telegram_id)
            return True
        except DoesNotExist:
            return False
