import re, random
from colorama import Fore, init

init(autoreset=True)

destinations = {
    "beaches": ["Bali", "Maldives", "Phuket"],
    "mountains": ["Swiss Alps", "Rocky Mountains", "Himalayas"],
    "cities": ["Tokyo", "Paris", "New York"],
}
jokes = [
    "Why don't programmers like nature? Too many bugs!",
    "Why did the computer go to the doctor? Because it had a virus!",
    "Why do travelers always feel warm? Because of all their hot spots!",
    "What do you call a pony with a cough? A little horse.",
    "Why did the robber jump in the shower? He wanted to make a clean getaway.",
]

EXIT_WORDS = {"exit", "bye", "quit"}


def normalize_input(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def recommend():
    while True:
        print(Fore.CYAN + "TravelBot: Beaches, mountains, or cities?")
        preference = normalize_input(input(Fore.YELLOW + "You: "))

        if preference in EXIT_WORDS:
            return "exit"

        if preference not in destinations:
            print(Fore.RED + "TravelBot: Sorry, I don't have that type of destination.")
            continue

        suggestion = random.choice(destinations[preference])
        print(Fore.GREEN + f"TravelBot: How about {suggestion}?")
        print(Fore.CYAN + "TravelBot: Do you like it? (yes/no)")
        answer = normalize_input(input(Fore.YELLOW + "You: "))

        if answer == "yes":
            print(Fore.GREEN + f"TravelBot: Awesome! Enjoy {suggestion}!")
            return
        elif answer == "no":
            print(Fore.RED + "TravelBot: Let's try another.")
            continue
        elif answer in EXIT_WORDS:
            return "exit"
        else:
            print(Fore.RED + "TravelBot: I'll suggest again.")
            continue


def packing_tips():
    print(Fore.CYAN + "TravelBot: Where to?")
    location = normalize_input(input(Fore.YELLOW + "You: "))
    if location in EXIT_WORDS:
        return "exit"

    print(Fore.CYAN + "TravelBot: How many days?")
    days = normalize_input(input(Fore.YELLOW + "You: "))
    if days in EXIT_WORDS:
        return "exit"

    print(Fore.GREEN + f"TravelBot: Packing tips for {days} days in {location}:")
    print(Fore.GREEN + "- Pack versatile clothes.")
    print(Fore.GREEN + "- Bring chargers/adapters.")
    print(Fore.GREEN + "- Check the weather forecast.")


def tell_joke():
    print(Fore.YELLOW + f"TravelBot: {random.choice(jokes)}")


def show_help():
    print(Fore.MAGENTA + "\nI can:")
    print(Fore.GREEN + "- Suggest travel spots (say 'recommendation')")
    print(Fore.GREEN + "- Offer packing tips (say 'packing')")
    print(Fore.GREEN + "- Tell a joke (say 'joke')")
    print(Fore.CYAN + "Type 'exit' or 'bye' to end.\n")


def chat():
    print(Fore.CYAN + "Hello! I'm TravelBot.")
    name = input(Fore.YELLOW + "Your name? ")
    print(Fore.GREEN + f"Nice to meet you, {name}!")
    show_help()

    while True:
        user_input = normalize_input(input(Fore.YELLOW + f"{name}: "))

        if "recommend" in user_input or "suggest" in user_input:
            if recommend() == "exit":
                break
        elif "pack" in user_input:
            if packing_tips() == "exit":
                break
        elif "joke" in user_input or "funny" in user_input:
            tell_joke()
        elif "help" in user_input:
            show_help()
        elif user_input in EXIT_WORDS:
            break
        else:
            print(Fore.RED + "TravelBot: Could you rephrase?")

    print(Fore.CYAN + "TravelBot: Safe travels! Goodbye!")


if __name__ == "__main__":
    chat()
