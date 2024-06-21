def print_and_log(file, text='', end='\n'):
    print(text, end=end)

    file.write(f"{text}{end}")
