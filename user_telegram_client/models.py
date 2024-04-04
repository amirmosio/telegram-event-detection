from peewee import Model, AutoField, CharField, OperationalError, ForeignKeyField, SqliteDatabase
db = SqliteDatabase('peewee_database')
db.connect()

class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    memid = AutoField()  # Auto-incrementing primary key.
    username = CharField()
    first_name = CharField()
    last_name = CharField()
    telegram_id = CharField()

    class Meta:
        table_name = 'user'

class UserCategorieMap(BaseModel):
    memid = AutoField()  # Auto-incrementing primary key.
    user = ForeignKeyField(User, backref='categories', lazy_load=False)
    categories = CharField()

    class Meta:
        table_name = 'user_category'

class UserGroupMap(BaseModel):
    memid = AutoField()  # Auto-incrementing primary key.
    user = ForeignKeyField(User, backref='groups', lazy_load=False)
    group_link = CharField()

    class Meta:
        table_name = 'user_group'

