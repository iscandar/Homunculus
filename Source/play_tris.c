#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "homunculus.h"
homunculus_brain* brain;

char gridChar(int i) {
    switch(i) {
        case -1:
            return 'O';
        case 0:
            return ' ';
        case 1:
            return 'X';
    }
}

void draw(int b[9]) {
    printf(" %c | %c | %c\n",gridChar(b[0]),gridChar(b[1]),gridChar(b[2]));
    printf("---+---+---\n");
    printf(" %c | %c | %c\n",gridChar(b[3]),gridChar(b[4]),gridChar(b[5]));
    printf("---+---+---\n");
    printf(" %c | %c | %c\n",gridChar(b[6]),gridChar(b[7]),gridChar(b[8]));
}

int win(const int board[9]) {
    //determines if a player has won, returns 0 otherwise.
    unsigned wins[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    int i;
    for(i = 0; i < 8; ++i) {
        if(board[wins[i][0]] != 0 &&
           board[wins[i][0]] == board[wins[i][1]] &&
           board[wins[i][0]] == board[wins[i][2]])
            return board[wins[i][2]];
    }
    return 0;
}


void computerMove(int board[9]) {

    float max=0 ;
    float* temp;
    int i,move=-1;
    for(i = 0; i < 9 ; i++)
    {
        if(board[i]==0)
        {
            board[i]=1;
            temp=run_brain(brain,&board,1);
            if(temp[0] > max)
            {
                max=temp[0];
                move=i;
            }
            board[i]=0;
        }
    }
    board[move]=1;
}

void playerMove(int board[9]) {
int move = 0;
do {
start:
printf("\nInput move ([0..8]): ");
scanf("%d", &move);
if(board[move] != 0) {
printf("Its Already Occupied !");
goto start;
}
printf("\n");
} while (move >= 9 || move < 0 && board[move] == 0);
board[move] = -1;
}

int main() {
    char file1[]="example/brain_created/brain_tris.data";
    brain=load_setting(file1);
    int board[9] = {0,0,0,0,0,0,0,0,0};
    //computer squares are 1, player squares are -1.
    printf("Computer: X, You: O\nPlay (1)st or (2)nd? ");
    int player=0;
    scanf("%d",&player);
    printf("\n");
    unsigned turn;
    for(turn = 0; turn < 9 && win(board) == 0; ++turn) {
        if((turn+player) % 2 == 0)
            computerMove(board);
        else {
            draw(board);
            playerMove(board);
        }
    }
    switch(win(board)) {
        case 0:
            printf("A draw. How droll.\n");
            break;
        case 1:
            draw(board);
            printf("You lose.\n");
            break;
        case -1:
            printf("You win. Inconceivable!\n");
            break;
    }
}

