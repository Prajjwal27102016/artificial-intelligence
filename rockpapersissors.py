import random
from colorama import init, Fore, Style

init(autoreset=True)

CHOICES = {
    'R': 'Rock',
    'P': 'Paper',
    'S': 'Scissors'
}


def player_choice():
    while True:
        choice = input(
            Fore.GREEN + "Choose Rock (R), Paper (P), or Scissors (S): " + Style.RESET_ALL
        ).strip().upper()

        if choice in CHOICES:
            return choice

        print(Fore.RED + "Invalid choice. Please enter R, P, or S." + Style.RESET_ALL)


def ai_choice():
    return random.choice(['R', 'P', 'S'])


def determine_winner(player_move, ai_move):
    if player_move == ai_move:
        return 'tie'

    winning_moves = {
        'R': 'S',
        'S': 'P',
        'P': 'R'
    }

    if winning_moves[player_move] == ai_move:
        return 'player'
    return 'ai'


def rock_paper_scissors():
    print(Fore.CYAN + "Welcome to Rock, Paper, Scissors!" + Style.RESET_ALL)
    player_name = input(Fore.GREEN + "Enter your name: " + Style.RESET_ALL).strip() or "Player"

    while True:
        player = player_choice()
        computer = ai_choice()

        print(f"{player_name} chose: {Fore.YELLOW}{CHOICES[player]}{Style.RESET_ALL}")
        print(f"Computer chose: {Fore.BLUE}{CHOICES[computer]}{Style.RESET_ALL}")

        result = determine_winner(player, computer)

        if result == 'tie':
            print(Fore.YELLOW + "It's a tie!" + Style.RESET_ALL)
        elif result == 'player':
            print(Fore.GREEN + f"Congratulations! {player_name} wins!" + Style.RESET_ALL)
        else:
            print(Fore.RED + "Computer wins!" + Style.RESET_ALL)

        play_again = input("Do you want to play again? (yes/no): ").strip().lower()
        if play_again != 'yes':
            print(Fore.CYAN + "Thank you for playing!" + Style.RESET_ALL)
            break


if __name__ == "__main__":
    rock_paper_scissors()