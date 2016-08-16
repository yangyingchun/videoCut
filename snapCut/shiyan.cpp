#include <iostream>

using namespace std;

class A{
public:
	A(){
		cout << "A的构造函数" << endl;
	}
	virtual ~A()
	{
		cout << "A的析构函数" << endl;
	}
};
class B : public A{
public:
	B(){
		cout << "B的构造函数" << endl;
	}
	~B(){
		cout << "B的析构函数" << endl;
	}
};
int* p, *q;
int* add(int n);
char* retChar();
int main1(void)
{
	/*char arr[] = {1,2,3,4,5,8,6,7,8,9};
	char* str = arr;
	cout << sizeof(str) << endl;
	cout << sizeof(arr) << endl;
	cout << strlen(str) << endl;*/
	/*int a = 0;
	cout << add(a) << endl;*/
	/*char* c;
	c = "welcome to hust!";
	cout << sizeof("welcome to hust!") << endl;
	cout << strlen("welcome to hust!") << endl;*/
	/*char c[] = "welcome to hust!";
	cout << sizeof(c) << endl;
	cout << strlen(c) << endl;*/
	//c[3] = 'y';
	//cout << c << endl;
	//int i, j;
	//cout << boolalpha << (i = j = 0) << endl;
	//cout << boolalpha << (i = j = 1) << endl;//表达式i= j = 0返回的是i的值
	//char *p;
	B* a = new B();
	A* b = a;
	delete b;
	/*int* m = add(4);
	cout << p << endl;
	cout << q << endl;*/
	//cout << a << endl;
	//cout << retChar() << endl;
	return 0;


}

int* add(int n)
{
	static int a = n;//局部静态变量虽然在函数之外不能使用但是分配的内存仍然在
    p = &a;
	q = &n;
	cout << p << endl;
	cout << q << endl;
	return p;//return 也是作为表达式的一部分
}

char* retChar()
{
	char* c = new char[2];
	//char* c = "yyc";
	c[0] = '1';
	c[1] = '\0';
	//free(c);
	char* p = c;
	//free(c);
	return p;
}