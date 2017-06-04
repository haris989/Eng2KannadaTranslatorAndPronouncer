my_list = [1,2,3,4,77]
max_value = max(my_list)
max_index = my_list.index(max_value)
print(max_index)

def index_of_max(list):
    max_value = max(list)
    max_index = list.index(max_value)
    return max_index



print(index_of_max(my_list))