from multiprocessing import Process, Queue

def f(q):
    # q.put([42, None, 'hello'])
    while not q.empty():
        print(q.get())

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    for i in range(10):
        q.put(i)
    # print q.get()    # prints "[42, None, 'hello']"
    p.join()
