# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def json_length():
    domain = 'Medicine'

    in_path = './data/corpus/' + domain + '/pdf_parses_10.jsonl'
    with open(in_path, 'r', encoding='utf-8') as papers:
        print(len(papers.readlines()))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def main():
    json_length()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
