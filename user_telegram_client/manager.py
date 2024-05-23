from peewee import SqliteDatabase, OperationalError, DoesNotExist
from .models import User, UserCategorieMap, UserGroupMap, GroupLinkIdMap


class QueryManager:
    def __init__(self) -> None:
        self.db = SqliteDatabase("peewee_database.db")
        self.db.connect()
        for model in [User, UserCategorieMap, UserGroupMap, GroupLinkIdMap]:
            try:
                self.db.create_tables([model])
            except OperationalError:
                print("Table already exists!")

    def check_if_user_has_been_here_before(self, telegram_id):
        try:
            _ = User.get(User.telegram_id == telegram_id)
            return True
        except DoesNotExist:
            return False

    def create_user(self, telegram_id, username, first_name, last_name):
        try:
            _ = User.create(
                telegram_id=telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                state=User.State.NOT_REGISTERED,
            )
            return True
        except DoesNotExist:
            return False

    def update_user_state(self, telegram_id, state):
        try:
            query = User.update(state=state).where(User.telegram_id == telegram_id)
            query.execute()
            return True
        except DoesNotExist:
            return False

    def get_user_state(self, telegram_id):
        try:
            user = User.get(User.telegram_id == telegram_id)
            return user.state
        except DoesNotExist:
            return None

    def update_group_for_user(self, telegram_id, groups):
        try:
            user = User.get(User.telegram_id == telegram_id)
            q = UserGroupMap.delete().where(UserGroupMap.user == user)
            q.execute()
            _ = UserGroupMap.bulk_create(
                [UserGroupMap(user=user, group_id=g) for g in groups]
            )
            return True
        except DoesNotExist:
            return False

    def update_topics_for_user(self, telegram_id, topics):
        try:
            user = User.get(User.telegram_id == telegram_id)
            q = UserCategorieMap.delete().where(UserCategorieMap.user == user)
            q.execute()
            _ = UserCategorieMap.bulk_create(
                [UserCategorieMap(user=user, category=t) for t in topics]
            )
            return True
        except DoesNotExist:
            return False

    def get_users_interested_in_topic_and_group(self, group=None, topic=None):
        try:
            if topic and group:
                users_q = (
                    User.select(User.telegram_id)
                    .distinct()
                    .join(UserCategorieMap)
                    .switch(User)
                    .join(UserGroupMap)
                    .where(
                        User.state == User.State.TOPIC_SET
                        and UserCategorieMap.category == topic
                        and UserGroupMap.group_id == group
                    )
                )
            else:
                users_q = User.select(User.telegram_id).where(
                    User.state == User.State.TOPIC_SET
                )
            return [u.telegram_id for u in users_q]
        except DoesNotExist:
            return False
