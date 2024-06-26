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
    username = CharField(null=True)
    first_name = CharField(null=True)
    last_name = CharField(null=True)
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
    group_id = CharField()

    class Meta:
        table_name = "user_group"


class GroupLinkIdMap(BaseModel):
    memid = AutoField()  # Auto-incrementing primary key.
    group_link = CharField()
    group_id = CharField()

    class Meta:
        table_name = "group_link_id_map"
