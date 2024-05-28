from enum import Enum

TopicClasses = {
    1: "1. Project/assignment/homework",
    2: "2. Exam/oral exam/mid term",
    3: "3. Studyplan",
    4: "4. Deadline/important dates",
    5: "5. Grades/marks/results",
    6: "6. Materials/recordings",
    7: "7. class information/class sessions",
    8: "8. Other",
}


class Messages:
    WELCOMING = (
        """Hi, This bot is going to help you manage your telegram groups and messages better."""
        + """\nJust tell us which groups and topics you are more interested in, and we will send you the conversation when someone send such message. For more information [Here](https://github.com/amirmosio/telegram-event-detection) is the repo of the project."""
        + """\nSend a "start" if you want to begin."""
        + """\nRemeber that you can always use "group" or "topic" to update your preferences."""
    )

    SEND_GROUPS_LINK = (
        """Send us the link to the groups which you are most interested in from Polimi Groups in different lines"""
        + """\nFor Example:"""
        + """\n<group-link1>"""
        + """\n<group-link2>"""
        + """\n<group-link3>"""
        + """\n..."""
    )

    SEND_INTERESTED_TOPICS = (
        "Send us the number of topics you are most interested in different lines from:"
        + "\n"
        + "\n".join([t for t in TopicClasses.values()])
        + "\n\nFor Example:"
        + "\n<topic-number1>"
        + "\n<topic-number2>"
        + "\n..."
    )

    FINAL = """You are all set up. From now on, if we see a new conversation is starting which is in your interests, we'll send you a notification."""

    INVALID_INPUT = """The input does not seems to be valid. If you want to update your prederences just use "group" or "topic"."""
    DONE = "All set"

    MESSAGE_NOTIFICATION_TEMPLATE = (
        "Group: {group}\nTopic: {topic}\n-----------------------------\n{message}"
    )


class Commands:
    Start = "start"
    Groups = "group"
    Topics = "topic"


class Patterns:
    Link = r"^https:\/\/(www\.)?t.me\/.*$"
    TopicIds = r"^[1-8]$"
