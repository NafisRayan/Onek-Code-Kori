#A python program using PyQt5 to make a simple GUI app.

import PyQt5.QtWidgets as pwt
import PyQt5.QtGui as pgi

class DataBase(pwt.QWidget):
    def __init__(self):

        self.userdata={}
        super().__init__()
        
        self.setWindowTitle("DataBase App")

        self.setLayout(pwt.QVBoxLayout())
        

        line= pwt.QLabel('Enter owner name: ')
        line.setFont(pgi.QFont('Helvetica',15))
        self.layout().addWidget(line)

        self.owner= pwt.QLineEdit()
        self.layout().addWidget(self.owner)

        tap0= pwt.QPushButton("Start application", clicked= lambda: mainf0())
        self.layout().addWidget(tap0)


        def mainf0():

            if len(self.owner.text())>0:
                print(F'Welcome {self.owner}')
                
                line= pwt.QLabel(f"Hi! I'm {self.owner.text()}.\nWelcome to my data collection app.\n")
                line.setFont(pgi.QFont('Helvetica',14))
                self.layout().addWidget(line)



                line2 = pwt.QLabel(f"Please enter your information.")
                line2.setFont(pgi.QFont('Helvetica',17))
                self.layout().addWidget(line2)

                line3 = pwt.QLabel("Format: name, age, income\n")
                line3.setFont(pgi.QFont('Helvetica',7))
                self.layout().addWidget(line3)
            
                data= pwt.QLineEdit()
                self.layout().addWidget(data)

                tap= pwt.QPushButton("Add User", clicked= lambda: mainf())
                self.layout().addWidget(tap)

                def jhamela():
                    lineU = pwt.QLabel(f"Invalid Input!!\nPlease follow the format: name, age, income")
                    lineU.setFont(pgi.QFont('Helvetica',15))
                    self.layout().addWidget(lineU)


                def mainf():
                    if ',' in data.text() and len(data.text())>0: 
                        x=data.text()
                        x=x.split(',')
                        name=x[0]
                        if x[1].replace(' ','').isnumeric() and x[2].replace(' ','').isnumeric():
                            age=int(x[1].replace(' ',''))
                            income=int(x[2].replace(' ',''))
                            
                            self.userdata[name]=(age,income)
                            lineU = pwt.QLabel(f"Hi {name}!\nYou have been added as a new user.\nYou are {age} years old and your income is {income} taka.\n\nDataBase: {self.userdata}.")
                            lineU.setFont(pgi.QFont('Helvetica',10))
                            self.layout().addWidget(lineU)
                            data.setText('')
                        else:
                            jhamela()
                    else:
                            jhamela()
            else:
                line2 = pwt.QLabel(f"No owner name was passed.\n!! YOU WON'T BE ABLE TO START THE APP WITHOUT A 'OWNER NAME !!")
                line2.setFont(pgi.QFont('Helvetica',10))
                self.layout().addWidget(line2)


        self.show()


app = pwt.QApplication([])
mw = DataBase()

app.exec_()
print(mw.userdata)

out = open('data er gushti.txt','w')

print(mw.userdata,file = out)