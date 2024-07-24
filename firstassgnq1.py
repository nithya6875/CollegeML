def count_no_vowels_consonants(str) :
    vowels = 0
    consonants = 0
    for i in str:
        if i in 'aeiouAEIOU':
            vowels += 1
        else:
            consonants += 1
    return vowels, consonants

def main():
    str = input("Enter a string: ")
    vowels, consonants = count_no_vowels_consonants(str)
    print("Vowels: ", vowels)
    print("Consonants: ", consonants)

if __name__ == "__main__":
    main()