from peewee import (
    Model,
    AutoField,
    CharField,
    OperationalError,
    ForeignKeyField,
    SqliteDatabase,
    Check,
)

db = SqliteDatabase("peewee_database")
db.connect()


class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    class State:
        NOT_REGISTERED = "NOT_REGISTERED"
        REGISTERED = "REGISTERED"
        GROUPS_SET = "GROUPS_SET"
        TOPIC_SET = "TOPIC_SET"  # everything set up
        WAIT_GROUP_SET = "WAIT_GROUP_SET"
        WAIT_TOPIC_SET = "WAIT_TOPIC_SET"

    memid = AutoField()  # Auto-incrementing primary key.
    username = CharField()
    first_name = CharField()
    last_name = CharField()
    telegram_id = CharField()
    state = CharField()

    class Meta:
        table_name = "user"


class UserCategorieMap(BaseModel):
    memid = AutoField()  # Auto-incrementing primary key.
    user = ForeignKeyField(User, backref="categories", lazy_load=False)
    category = CharField()

    class Meta:
        table_name = "user_category"


class UserGroupMap(BaseModel):
    memid = AutoField()  # Auto-incrementing primary key.
    user = ForeignKeyField(User, backref="groups", lazy_load=False)
    group_link = CharField()

    class Meta:
        table_name = "user_group"
