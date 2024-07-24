'''
Write a program to find the number of
 common elements betweeen two lists.
The lists contain integers.
'''

def common_elements(list1, list2):
    common = []
    for i in list1:
        if i in list2:
            common.append(i)
    return common

def main():
    list1 = [1, 2, 3, 4, 5]
    list2 = [4, 5, 6, 7, 8]
    common = common_elements(list1, list2)
    print(common)

if __name__ == "__main__":
    main()