from tqdm import tqdm
import queue
import time
import threading
import translators as ts
from translators.server import TranslatorError
from requests.exceptions import ConnectionError as ConnectionE


def _process_text(message):
    message_google = ts.translate_text(
        query_text=message,
        translator="google",
        from_language="auto",
        to_language="en",
    )
    return message_google


def translate_messages(df_original):

    translation_text_google = list(df_original["text"].array)
    df_translated = df_original.copy()

    items_queue = queue.Queue()
    running = False

    def items_queue_worker():
        while running:
            try:
                item = items_queue.get(timeout=0.01)
                if item[0] is None:
                    continue
                retries = 0
                while True:
                    try:
                        t_message = _process_text(item[0])
                        translation_text_google[item[1]] = t_message
                        progress_text_bar.update(1)
                        progress_text_bar.refresh()
                        break
                    except TranslatorError:
                        break
                    except IndexError:
                        break  # It it appears when there is a link in the message
                    except ConnectionE as e1:
                        retries += 1
                        time.sleep(retries * 3)
                    except Exception as ee:
                        retries += 1
                        time.sleep(retries * 3)
                        print("ee:", str(ee))
                items_queue.task_done()
            except queue.Empty:
                pass

    running = True
    worker_threads = 5
    for _ in range(worker_threads):
        threading.Thread(target=items_queue_worker).start()

    progress_text_bar = tqdm(range(len(df_translated)))
    progress_text_bar.update(0)
    progress_text_bar.refresh()

    for i in range(len(df_translated)):
        items_queue.put((df_translated["text"].iloc[i], i))

    items_queue.join()
    running = False

    df_translated["text"] = translation_text_google
    return df_translated
