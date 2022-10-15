import heapq
import itertools


class PriorityQueue(object):
    """
    PriorityQueue with O(1) update and deletion of objects
    """

    def __init__(self, initial=None, priorities=None):

        if initial is None:
            initial = []
        if priorities is None:
            initial = []

        self.pq = []
        self.entry_finder = {}  # mapping of tasks to entries
        self.removed = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

        assert (len(initial) == len(priorities))
        for i in range(len(initial)):
            self.push(initial[i], priority=priorities[i])

    def push(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.delete(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def delete(self, task):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.removed

    def remove_all_tasks_of_type(self, type):
        """Removes all existing tasks of a specific type"""
        keys = list(self.entry_finder.keys())
        for event in keys:
            u, type_, v = event
            if type_ == type:
                self.delete(event)

    def pop_priority(self):
        """
        Remove and return the lowest priority task with its priority value.
        Raise KeyError if empty.
        """
        while self.pq:
            priority, _, task = heapq.heappop(self.pq)
            if task is not self.removed:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def pop(self):
        """
        Remove and return the lowest priority task. Raise KeyError if empty.
        """
        task, _ = self.pop_priority()
        return task

    def priority(self, task):
        """Return priority of task"""
        if task in self.entry_finder:
            return self.entry_finder[task][0]
        else:
            raise KeyError('task not in queue')

    def find(self, task):
        """Return True if task is in the queue"""
        return task in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)

    def __str__(self):
        return str(self.pq)

    def __repr__(self):
        return repr(self.pq)

    def __setitem__(self, task, priority):
        self.push(task, priority=priority)

    def __iter__(self):
        return iter(self.pq)