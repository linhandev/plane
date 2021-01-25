import multiprocessing as mp
import time

def foo(q):
    while 1:
        print(q.get())

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    count = 0
    while True:
        time.sleep(1)
        q.put(count)
        count += 1
    p.join()
