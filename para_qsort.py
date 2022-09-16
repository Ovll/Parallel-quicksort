from sys import argv
from random import randint

def main():
    
    try:
        if len(argv) > 1:
            print(f'Opening {argv[1]}')
            data = list(map(int, open(str(argv[1]))))
        else:
            with open("random_data.txt", "w") as f:
                f.writelines((str(randint(-5000,5000)) + '\n' for _ in range(2_000_000)))
            print("Opening random_data.txt")
            data = list(map(int, open("random_data.txt")))
    except:
        print(f"Can't open {argv[1]}")
        exit()
    print(len(data))

if __name__=="__main__":
    main()