#include <Stepper.h>
#include <String.h>
// -------------------常数---------------------
#define ONESTEP 205 //魔方顺时针转90度
#define RONESTEP -205
#define stepsPerRevolution 205 // 步进电机旋转一圈是多少步
#define stepperSpeed 70 //电机转速,1ms一步
#define timeBetweenMoves  50  // ms    between U2 R2 or U R etc.
#define timeBetweenComs  4000 // ms    between commands
const int _angle[4]={0,ONESTEP/4,2*ONESTEP/4,3*ONESTEP/4};
// --------------------------------------------
// -------------------变量---------------------
// 设置步进电机的步数和引脚
// 上黄右橘前绿下白左红后蓝
Stepper FF(stepsPerRevolution,28,29,30,31);
Stepper DD(stepsPerRevolution,22,23,24,25);
Stepper BB(stepsPerRevolution,34,35,36,37);
Stepper LL(stepsPerRevolution,38,39,40,41);
Stepper UU(stepsPerRevolution,44,45,46,47);
Stepper RR(stepsPerRevolution,50,51,52,53);
String solution = "";
// -------------------------------------------
void getSolution();
void Solve();

void setup()
{
  // 设置转速，单位r/min
  FF.setSpeed(stepperSpeed);
  UU.setSpeed(stepperSpeed);
  RR.setSpeed(stepperSpeed);
  LL.setSpeed(stepperSpeed);
  DD.setSpeed(stepperSpeed);
  BB.setSpeed(stepperSpeed);
  pinMode(13,OUTPUT);
  // 初始化串口
  Serial.begin(9600);
}


void loop()
{
  getSolution();
  if(solution.length()>0) Solve();
  delay(timeBetweenComs);
}


void getSolution(){
  while (Serial.available() > 0) {   
    solution += char(Serial.read()); 
  }
  Serial.print("get solution: ");
  Serial.println(solution);
  // 格式：solution = "U2F2L2D2B2U2F2L2F2R2F2U1L2R2F2U3B2U1L2U3R1U3D2F2R1D2U1F1U1B1R3L2U3";
}


void Solve(){
  int com_len=solution.length();
  Serial.print("command length:");
  Serial.println(com_len);
    for(int i = 0; i < com_len; i += 2)
    {
        char ch = solution[i];
        int n = solution[i + 1] - '0';
        switch(ch)
        {
            case 'F':FF.step(_angle[n]); for(int i=FF.motor_pin_1;i<FF.motor_pin_1+4;i++) digitalWrite(i,LOW); break;
            case 'U':UU.step(_angle[n]); for(int i=UU.motor_pin_1;i<UU.motor_pin_1+4;i++) digitalWrite(i,LOW); break;
            case 'L':LL.step(_angle[n]); for(int i=LL.motor_pin_1;i<LL.motor_pin_1+4;i++) digitalWrite(i,LOW); break;
            case 'R':RR.step(_angle[n]); for(int i=RR.motor_pin_1;i<RR.motor_pin_1+4;i++) digitalWrite(i,LOW); break;
            case 'D':DD.step(_angle[n]); for(int i=DD.motor_pin_1;i<DD.motor_pin_1+4;i++) digitalWrite(i,LOW); break;
            case 'B':BB.step(_angle[n]); for(int i=BB.motor_pin_1;i<BB.motor_pin_1+4;i++) digitalWrite(i,LOW); break;            
        }
        // Turning off the digital output of every pin after steps are made is crutial; otherwise, the current will be too large. 
        // To do so, motor_pin_i in Stepper.h must be removed from private to public. 
        digitalWrite(13,HIGH);
        delay(timeBetweenMoves);
        digitalWrite(13,LOW);
        Serial.print(ch);
        Serial.println(n);
    }
  solution = "";
  Serial.println("empty now");
} 
