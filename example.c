#include<stdio.h>

int main(int argc, char** argv){
int i  = 0;
LOOP:
  printf("%d\n",i);
  if(i == 5) goto END;
  i++;
  goto LOOP;
END:
  printf("end!\n");
  return 0;
}
