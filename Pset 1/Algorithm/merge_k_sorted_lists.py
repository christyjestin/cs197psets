from typing import List, Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next


def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    head = None
    tail = None
    # remove empty lists
    lists = list(filter(lambda x: x, lists))
    while lists:
        # pop smallest value
        min_index, _ = min(list(enumerate(lists)), key = lambda x: x[1].val)
        min_val = lists[min_index].val
        if lists[min_index].next:
            lists[min_index] = lists[min_index].next
        else:
            lists.pop(min_index)
        # add value onto linked list
        if head is None:
            head = ListNode(val = min_val)
            tail = head
        else:
            tail.next = ListNode(val = min_val)
            tail = tail.next
    return head


# recursively construct linked list
def create_linked_list(lst: List[int]) -> ListNode:
    return ListNode(val = lst[0], next = create_linked_list(lst[1:])) if lst else None


# process list of lists to create list of ListNodes
def process_lists(lists: List[List[int]]) -> List[ListNode]:
    return [create_linked_list(lst) for lst in lists]


# convert linked list to standard list using head
def create_regular_list(linked_list: ListNode) -> List[int]:
    output = []
    head = linked_list
    while head:
        output.append(head.val)
        head = head.next
    return output


def run_tests(tests, expected):
    output = [create_regular_list(merge_k_sorted_lists(process_lists(test))) for test in tests]
    if len(output) != len(expected):
        print(f"Tests of length {len(tests)} are incompatible with expected answers \
                of length {len(expected)}")
        return
    print("Accepted" if output == expected else "Rejected")
    print("Output: ")
    for response in output:
        print(response)
    print("Expected: ")
    for answer in expected:
        print(answer)


test_cases = [[[1, 4, 5], [1, 3, 4], [2, 6]],
              [],
              [[]]]
answers = [[1, 1, 2, 3, 4, 4, 5, 6],
           [],
           []]

run_tests(test_cases, answers)
