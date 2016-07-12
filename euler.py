import math
from math import sqrt
from collections import defaultdict
from collections import OrderedDict
import operator
import datetime
import itertools
import os
import sys
import numpy
import time
import re
import random

SCREENSCALE = 64

# Utility functions ------------------------------------------------------------------------------------
	
class Prime():
	
	ordered = (2,3)
	unordered = frozenset(ordered)
	MillerRabinTrials = 6
	factorcache = dict()
	maxCachedPrime = 3
	
	#http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188
	@staticmethod
	def primesfrom2to(n):
		""" Input n>=6, Returns a array of primes, 2 <= p < n """
		sieve = numpy.ones(n/3 + (n%6==2), dtype=numpy.bool)
		for i in xrange(1,int(n**0.5)/3+1):
			if sieve[i]:
				k=3*i+1|1
				sieve[       k*k/3     ::2*k] = False
				sieve[k*(k-2*(i&1)+4)/3::2*k] = False
		return numpy.r_[2,3,((3*numpy.nonzero(sieve)[0][1:]+1)|1)]
	
	@staticmethod
	def nthPrimeTerm(n):
		
		n = n-1
		if n < 0:
			print 'A negative prime term?  Really?'
			return -1
		if len(Prime.ordered) < n: 
			print 'This is larger than the prime cache.'
			return -1
			
		return Prime.ordered[n]
	
	@staticmethod
	def warmup(n):
		print 'Warming up prime list.'
		Prime.ordered = tuple([long(prime) for prime in Prime.primesfrom2to(n)])
		Prime.unordered = frozenset(Prime.ordered)

		Prime.maxCachedPrime = n
		
		with open('primes.txt', 'w') as f:
			for prime in Prime.ordered:
				f.write(str(prime) + '\n')
		
		print 'Prime list created.  Largest term:', Prime.ordered[-1]
		
	@staticmethod
	def isPrime(n):
		if n > Prime.maxCachedPrime:
			return Prime.isPrime_MillerRabin(n)
		else:
			return n in Prime.unordered
		
	# Code is obtained from https://gist.github.com/bnlucas/5857478
	#
	# Assumed:
	# Copyright Nathan Lucas, 2013
	#
	# No license is provided with the source code, however I would gladly
	# update this if anyone is aware of how it should be licensed.
	
	@staticmethod
	def isPrime_MillerRabin(n):
	
		if n == 2:
			return True
		if not n & 1:
			return False

		def check(a, s, d, n):
			x = pow(a, d, n)
			if x == 1:
				return True
			for i in xrange(s - 1):
				if x == n - 1:
					return True
				x = pow(x, 2, n)
			return x == n - 1

		s = 0
		d = n - 1

		while d % 2 == 0:
			d >>= 1
			s += 1

		for i in xrange(Prime.MillerRabinTrials):
			a = random.randrange(2, n - 1)
			if not check(a, s, d, n):
				return False
		return True
		
		
	@staticmethod
	def gcd(a, b):	
    
		# Speed checks.
		aprime = Prime.isPrime(a)
		bprime = Prime.isPrime(b)
		
		if aprime and bprime: return 1
		
		if aprime:
			if b%a == 0: return a
			else:        return 1
			
		if bprime:
			if a%b == 0: return b
			else:        return 1
    
		factors_a = Prime.factors(a)
		factors_b = Prime.factors(b)
		
		if a in factors_b: return a
		if b in factors_a: return b
		
		for i in factors_a[::-1]:
			if i in factors_b:
				return i
				
		return 1
		
	@staticmethod
	def primerange(lower, upper):
		for prime in Prime.ordered:
			if prime < lower: continue
			if prime > upper: return
			yield prime
		

	@staticmethod
	def primefactors(n):
	
		if Prime.isPrime(n): return
		
		limit = int(sqrt(n))+1
		
		i = 2
		while i <= limit:
			if n % i == 0:
				yield i
				n = n / i
				limit = sqrt(n)   
			else:
				i += 1
		if n > 1:
			yield n
				
	@staticmethod	
	def collectFactors(factors, base):
		newFactors = set(factors)
	
		for factor in factors:
			if Prime.isPrime(factor): continue
			
			collected = None
			
			if factor in Prime.factorcache:
				collected = Prime.factorcache[factor]
			
			else:
				collected = Prime.factors(factor)
			
			for element in collected:
				if base%element == 0:
					newFactors.add(element)
					newFactors.add(base/element)
				
		if base in newFactors: newFactors.remove(base)
		return newFactors
				
		
	# http://stackoverflow.com/a/12273111
	@staticmethod
	def factors(n):
		if Prime.isPrime(n): return tuple()
		
		# Check and see if n is already cached.
		if n in Prime.factorcache: return Prime.factorcache[n]
			
		results = set()		
		for f in set(Prime.primefactors(n)):
			results.add(f)
			results.add(n/f)
			
		if n in results: results.remove(n)
		if 1 in results: results.remove(1)
		
		results.update(Prime.collectFactors(results, n))
		
		if n in results: results.remove(n)
		if 1 in results: results.remove(1)
		
		resultTuple = tuple(results)
		Prime.factorcache[n] = resultTuple		
		return resultTuple
		
	@staticmethod
	def relativelyPrimeTo(src, against):
		return Prime.gcd(src, against) == 1
		
def stringify(*args):
	return ' '.join([str(arg) for arg in args])
		
def maxDigitVal(i):
	return int('1'*i)*9
	
def strfrac(a_collection):
	return str(a_collection[0]) + '/' + str(a_collection[1])
	
def strfrac(a, b):
	return str(a) + '/' + str(b)
	
def stringPermutation(string1, string2):

	if len(string1) != len(string2): return False
	
	j = defaultdict(int)
	for elem in string1: j[elem] += 1
		
	k = defaultdict(int)
	for elem in string2: k[elem] += 1
		
	return j==k
	
def numericPermutation(int1, int2):

	j = defaultdict(int)
	k = defaultdict(int)
	
	while (int1 > 0):
		j[int1%10] += 1
		int1 /= 10
		
	while (int2 > 0):
		k[int2%10] += 1
		int2 /= 10
		
	return j==k
	
		
def isTrianglegon(n, returnTermNumber=False):
	term = (math.sqrt(8*n+1)-1)/2
	
	if not returnTermNumber:
		return int(term) == term
	return term

def isPentagon(n, returnTermNumber=False):
	term = (math.sqrt(24*n + 1)+1)/6
	
	if not returnTermNumber:
		return int(term) == term
	return term
			
def screenWidth():
	width, height = 80, 25
	return width
	
def isPalindrome(a):
	return a == a[::-1]

def mul(factors):
	return reduce(operator.mul, factors, 1)
	
def listsum(container):
	total = []
	for elem in container:
		total += elem
	return total
			
def formatListInto(flatlist, n):
	return [flatlist[i:i+n] for i in range(0, len(flatlist), n)]
	
def prettifyScreenScale(list, separator):

	screenlines = []
	
	currentLine = ''
	
	for elem in list:
		if len(currentLine) + len(str(elem)) + len(separator) > (screenWidth()-1):
			screenlines.append(currentLine)
			currentLine = str(elem) + separator
		else:
			currentLine += str(elem) + separator
			
	if currentLine != '':
		screenlines.append(currentLine)
		
	if screenlines[-1][-len(separator):] == separator:
		screenlines [-1] = screenlines[-1][0:-len(separator)]
		
	return screenlines
	
def FibGen():
	a, b = 1, 1
	yield a
	yield b
	
	while 1:
		nextTerm = a + b
		yield nextTerm
		a = b
		b = nextTerm
		
def fib_n(n):
	gen = FibGen()
	generatedTerm = 0
	
	for i in xrange(n):
		generatedTerm = gen.next()

	return generatedTerm
	
def perfectSquare(n):
	return int(math.sqrt(n))**2 == n
	
def sqrtExpand(n):
		
		# Check if term is a perfect square and quit early.
		terms = [int(math.sqrt(n))]
		if terms[0]**2 == n: return terms
		
		#https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Continued_fraction_expansion
	
		num = terms[0]
		denom = n - (num**2)
		
		first = (num, denom)
		terms.append((terms[0] + num) / denom)
		
		# Repeat until the numerator and denominator loop and match the first term.	
		while 1:
		
			# Each iteration, build the numerator.
			num = denom * terms[-1] - num
			denom = (n - num**2) / denom
			
			if first != (num, denom): 
				terms.append((terms[0] + num) / denom)
			else:
				break
		
		return terms
		
def rationalize(n, numTerms=None):

	#https://www.ocf.berkeley.edu/~wwu/cgi-bin/yabb/YaBB.cgi?board=riddles_medium;action=display;num=1080900827
	# For talking about n-2 and and n-1 terms, which enables the entire sequence to be calculated directly.

	expanded = sqrtExpand(n)
	period = len(expanded)-1
	terms = [(0, 1), (1, 0)]
	
	useTerms = numTerms
	if useTerms is None:
		if period % 2 == 0: useTerms = period
		else:               useTerms = 2 * period
			
	# Don't repeat the first term after using it the first time.
	# By using a shorter list, we can loop around, but need to remember that we have
	# looked at an existing entry outside of it.
	truncated = expanded[1:]
	
	# Solve the first term because we need to discard the constant in future things.
	num_m1, denom_m1 = terms[-1]
	num_m2, denom_m2 = terms[-2]
		
	terms.append((expanded[0]*num_m1 + num_m2, expanded[0]*denom_m1 + denom_m2))
	
	while len(terms) < useTerms+2:
		num_m1, denom_m1 = terms[-1]
		num_m2, denom_m2 = terms[-2]
		
		multiplier = truncated[(len(terms)-3) % period]
		
		terms.append((multiplier*num_m1 + num_m2, multiplier*denom_m1 + denom_m2))
			
	# Don't return the -2 and -1 terms.
	return terms[2:]
		
# Integers only.  No rounding is performed.
def dec_bits_n(top, bottom, afterDecimal, ignoreZero=False):
	value = ''
	
	decval = top/bottom
	value += str(decval) + '.'
	
	for i in xrange(1, afterDecimal+1):
		top = 10 * (top - ( decval * bottom ))
		decval = top/bottom
		value += str(decval)
		
	if ignoreZero:
		while value[-1] == '0':
			value = value[:-1]
	return value		
	
def list_chunks(l, s):
	for j in xrange(0, len(l), s):
		yield l[j:j+s]
		

def rotate_left(l, n):
	if len(l) == 1: return l	
	return l[-n:] + l[:-n]
	
def rotate_right(l, n):
	if len(l) == 1: return l	
	return l[n:] + l[:n]
	
def rotations(x):
	yield x
	z = rotate_left(x,1)
	
	while z != x:
		yield z
		z = rotate_left(z,1)
	
	
class EXPAND():
	def __init__(self, a):
		self.a = a
		
	def __str__(self):
		if type(self.a) is str:
			return self.a
			
		elif type(self.a) in [tuple, list, set, frozenset]:
			rstr = ''
			for q in self.a:
				if rstr != '': rstr += '\n'
				rstr += str(q)
			return rstr
				
		elif type(self.a) is dict:
			rstr = ''
			for q in self.a:
				if rstr != '': rstr += '\n'
				rstr += str(q) + ' - ' + str(self.a[q])
			return rstr
			
		return str(self.a)
		
		
	
# decorator shenanigans --------------------------------------------------------------------------------

def interpretDesc(string, **kwargs):

	def getCapturedVars(string):
	
		vars = dict()

		valid = True
		while valid:
				
			if string.count('%'):
			
				symbolPos = string.index('%')
				string = string[symbolPos:]

				if string.count('(') and string.count(')'):
				
					lparen = string.index('(')
					rparen = string.index(')')
					
					if lparen < rparen:
					
						captured = string[lparen+1:rparen]
						
						if captured not in kwargs: print "ERROR:", captured, "NOT IN", kwargs
						else:                      vars['%(' + captured + ')'] = kwargs[captured]						
						
						string = string[rparen+1:]
					
					else:
						string = string[lparen+1:]
									
				else: valid = False
			else: valid = False
				
		return vars
	
	vars = getCapturedVars(string)
	
	for var in vars:
		string = string.replace(var, str(vars[var]))	
		
	return string

			
def SetupAndTime(probNum, description, **kwargs):
	def real_decorator(func):
		def wrapper():
			print '\n//-----------//-Problem ' + str(probNum) + '-//----------//----------//\n'
			print 'Problem description:'
			print interpretDesc(description, **kwargs)
			print ''
		
			iterations = 0
			if '_iters' in kwargs:
				iterations = kwargs['_iters']
				del kwargs['_iters']

			if iterations > 0:
			
				print 'Running', iterations, 'iterations.'
				
				begin = time.clock()
				for i in xrange(iterations): func(**kwargs)
				end = time.clock()
				
				print "\nElapsed time (s):", end-begin, '\n'
				return end - begin
				
			else:

				begin = time.clock()
				resultstr =  func(**kwargs)
				end = time.clock()
				
				print resultstr
				print "\nElapsed time (s):", end-begin, '\n'
				return end - begin
				
		return wrapper
	return real_decorator
				
# Solving problems --------------------------------------------------------------------------------
	
@SetupAndTime(1,'Sum of numbers evenly divisible by %(divisby) below %(maxterm)', divisby=(3, 5), maxterm=1000)
def problem1(**kwargs):

	maxterm = kwargs['maxterm']
	divibility = kwargs['divisby']
	
	def isModulus(n, terms):
		for term in terms:
			if n%term == 0: return True
		return False
	
	return sum([t for t in xrange(1, maxterm+1) if isModulus(t, divibility)])

# -----------------------------
# -----------------------------

@SetupAndTime(2,'Sum of even-valued Fibonacci terms below %(upperbound)', upperbound=4000000)
def problem2(**kwargs):
	upperbound = kwargs['upperbound']
	
	gen = FibGen()
	gen.next()
	
	total = 0
	
	while 1:
		z = gen.next()
		if z > upperbound:
			break
		elif z%2 == 0:
			total += z

	return total
	
# -----------------------------
# -----------------------------

@SetupAndTime(3,'Largest prime factor of %(bignumber)', bignumber=600851475143)
def problem3(**kwargs):
	term = kwargs['bignumber']
	return max(Prime.primefactors(term))
	
# -----------------------------
# -----------------------------

@SetupAndTime(4,'Largest palindrome formed by multiplying two %(numDigits) digit numbers', numDigits=3)
def problem4(**kwargs):
		
	numDigits = kwargs['numDigits']
	maxterm = maxDigitVal(numDigits)
	
	largest = 0
	terms = (0,0)
	
	for i in xrange(maxterm, 0, -1):
		for j in xrange(i, 0, -1):
			product = i*j
			
			if product > largest and isPalindrome(str(product)):
				largest = product
				terms = (i, j)
	
	return stringify(terms, '->', largest)	
# -----------------------------
# -----------------------------

@SetupAndTime(5,'Smallest positive number evenly divisible by the first %(upto) natural numbers:', upto=20)
def problem5(**kwargs):

	# Largest term to look for?
	upto = kwargs['upto']
	
	primes = [factor for factor in xrange(1,upto+1) if Prime.isPrime(factor)]
	
	minterm = mul(primes)
		
	coverage = [z for z in xrange(upto, 1, -1) if not Prime.isPrime(z)]
	
	total = 0
	while 1:
		total += minterm
		
		valid = True
		
		for element in coverage:
			if total % element != 0:
				valid = False
				break
				
		if valid:
			return total
				
# -----------------------------
# -----------------------------

@SetupAndTime(6,'Difference between square sum and sum squared for [ 0 - %(maximum) ]:', maximum=100)
def problem6(**kwargs):

	maximum = kwargs['maximum']
	
	def sqrsum(maximum):
		return math.pow(sum(range(1,maximum+1)), 2)
		
	def sumsqr(maximum):
		return sum([x*x for x in range(1, maximum+1)])
	
	return sqrsum(maximum) - sumsqr(maximum)
	
# -----------------------------
# -----------------------------

@SetupAndTime(7,'The %(nthTerm)th prime term is:', nthTerm=10001)
def problem7(**kwargs):
	return Prime.nthPrimeTerm(kwargs['nthTerm'])
	
# -----------------------------
# -----------------------------

@SetupAndTime(8,'The %(numAdjacent) adjacent digits with the greatest product:', numAdjacent=13)
def problem8(**kwargs):

	windowsize = kwargs['numAdjacent']

	string =    '73167176531330624919225119674426574742355349194934' \
				'96983520312774506326239578318016984801869478851843' \
				'85861560789112949495459501737958331952853208805511' \
				'12540698747158523863050715693290963295227443043557' \
				'66896648950445244523161731856403098711121722383113' \
				'62229893423380308135336276614282806444486645238749' \
				'30358907296290491560440772390713810515859307960866' \
				'70172427121883998797908792274921901699720888093776' \
				'65727333001053367881220235421809751254540594752243' \
				'52584907711670556013604839586446706324415722155397' \
				'53697817977846174064955149290862569321978468622482' \
				'83972241375657056057490261407972968652414535100474' \
				'82166370484403199890008895243450658541227588666881' \
				'16427171479924442928230863465674813919123162824586' \
				'17866458359124566529476545682848912883142607690042' \
				'24219022671055626321111109370544217506941658960408' \
				'07198403850962455444362981230987879927244284909188' \
				'84580156166097919133875499200524063689912560717606' \
				'05886116467109405077541002256983155200055935729725' \
				'71636269561882670428252483600823257530420752963450'
				
	def largest(string, windowsize):
	
		greatest = 0
		term = ''
		
		for i in range(len(string) - windowsize):
			substr = string[i: i+windowsize]
			
			# Obviously anything multiplied by 0 is going to be zero.
			# Ignore anything that has a zero, and if we don't find
			# anything, we'll just return zero.
			
			if '0' in substr: continue
			
			multerms = mul([long(i) for i in substr])
			if multerms > greatest:
				term = substr
				greatest = multerms
				
		return (term, greatest)

	substr = largest(string, windowsize)
	
	return stringify(substr[0], '->', substr[1])
	
# -----------------------------
# -----------------------------

@SetupAndTime(9,'Maximized Pythagorean triplet where a+b+c=%(maxRange):', maxRange=1000)
def problem9(**kwargs):

	def isTriplet(a, b, c):
		return (a**2) + (b**2) == (c**2)
		
	maxRange = kwargs['maxRange']
	
	finalTerms = [-1, -1, -1]
		
	for aTerm in xrange(maxRange/3, 0, -1):
	
		found = False
		
		for bTerm in xrange((maxRange/2), aTerm-1, -1):
			cTerm = maxRange - aTerm - bTerm
			
			# Obviously if a+b <c, then a*a + b*b < c*c.
			# And as our terms decrease, this will happen for every remaining number.
			if cTerm > aTerm+bTerm: break
			
			if isTriplet(aTerm, bTerm, cTerm):
				finalTerms = (aTerm, bTerm, cTerm)
				found = True
				break
				
		if found:
			break
	
	return stringify(zip(('A:', 'B:', 'C:'), finalTerms), 'a*b*c=', mul(finalTerms))

# -----------------------------
# -----------------------------

@SetupAndTime(10,'Sum of primes below %(maxTerm):', maxTerm=2000000)
def problem10(**kwargs):
	
	maxterm = kwargs['maxTerm']
		
	total = 0
		
	for prime in Prime.ordered:
		if prime > maxterm:
			return total
		total += prime
		
	if maxterm > Prime.ordered[-1]:
		print 'Prime cache only contains primes up to', Prime.ordered[-1]
		
	return -1
	
# -----------------------------
# -----------------------------

@SetupAndTime(11,'Greatest product of %(numterms) adjacent numbers:', numterms=4)
def problem11(**kwargs):

	numterms = kwargs['numterms']
	
	matrix =['08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08',
			 '49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00',
			 '81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65',
			 '52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91',
			 '22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80',
			 '24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50',
			 '32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70',
			 '67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21',
			 '24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72',
			 '21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95',
			 '78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92',
			 '16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57',
			 '86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58',
			 '19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40',
			 '04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66',
			 '88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69',
			 '04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36',
			 '20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16',
			 '20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54',
			 '01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48']
			 
			 
	def searchDown(matrix, point, numterms):
		y, x = point[0], point[1]
		return [matrix[y+n][x] for n in range(numterms)]
		
	def searchDownLeft(matrix, point, numterms):
		y, x = point[0], point[1]
		return [matrix[y+n][x-n] for n in range(numterms)]
		
	def searchDownRight(matrix, point, numterms):
		y, x = point[0], point[1]
		return [matrix[y+n][x+n] for n in range(numterms)]
			 
			 
	# Reformat the matrix
	matrix = [ [int(element, 10) for element in string.split(' ')] for string in matrix]
			 
	width = len(matrix[0])
	height = len(matrix)
	
	
	maxterm = 0
	numbers = []
	startpos = None
	dir = None
	
	for y in xrange(0, height):
		for x in xrange(0, width):
			point = (y, x)
			combinations = []
			
			xleft = x >= numterms - 1
			xright = x <= width - numterms
			ydown = y <= height - numterms
			
			if ydown:
				combinations.append([searchDown(matrix, point, numterms), 'down', point])
			if xleft and ydown:
				combinations.append([searchDownLeft(matrix, point, numterms), 'down left', point])
			if xright and ydown:
				combinations.append([searchDownRight(matrix, point, numterms), 'down right', point])
				
			for terms in combinations:
				prod = mul(terms[0])
				if prod > maxterm:
					maxterm = prod
					numbers = terms[0]
					dir = terms[1]
					startpos = 'Starting position: X=' + str(terms[2][1]) + ' Y=' + str(terms[2][0])
		
	return stringify(maxterm, '--', 'Combination:', numbers, startpos, '('+dir+')')
			
# -----------------------------
# -----------------------------

# Project Euler's definition of divisors changes quite a bit.
# Sometimes it includes n and 1, and other times it doesn't.
# Our prime class generates without N and 1, so we just need
# to add 2 to the length of the factors at the end of this.

@SetupAndTime(12,'The first triangle term with %(numdivs) divisors:', numdivs=501)
def problem12(**kwargs):

	numDivisors = kwargs['numdivs']

	counter = 0
	triangleTerm = 0
	maxDivisors = 0
	
	# There has to be a smarter way to do this than brute forcing it.
	# But I don't know of how at the moment, and 6s is pretty good for now...
	
	print 'Searching for term with', numDivisors, 'divisors.'
	
	
	
	while 1:
	
		counter += 1
		triangleTerm += counter
		
		# As much as I hate ungeneric code in these, Project Euler is targeted at a certain solution.
		# It's not interested about small numbers much of the time -- and so we can look at ways to
		# handle optimization for larger numbers.
		
		if triangleTerm > 100 and (triangleTerm % 6 != 0): continue
		
		# Three seconds still sucks.  But as our final term is huge...we'll be at least pumping the factor
		# cache a bit for future problems...
		
		
		divisors = len(Prime.factors(triangleTerm))+2
		
		if divisors > maxDivisors:
			print ' - Term', counter, '(' + str(triangleTerm) + '):', divisors, 'divisors.'
			maxDivisors = divisors
		
		if divisors >= numDivisors:
			return stringify(divisors, "divisors for ", triangleTerm, "-- term", counter)	
			
# -----------------------------
# -----------------------------

@SetupAndTime(13,'First %(numdigits) digits in the sum of that giant hulking list of terms:', numdigits = 10)
def problem13(**kwargs):
	terms = [37107287533902102798797998220837590246510135740250,
			 46376937677490009712648124896970078050417018260538,
			 74324986199524741059474233309513058123726617309629,
			 91942213363574161572522430563301811072406154908250,
			 23067588207539346171171980310421047513778063246676,
			 89261670696623633820136378418383684178734361726757,
			 28112879812849979408065481931592621691275889832738,
			 44274228917432520321923589422876796487670272189318,
			 47451445736001306439091167216856844588711603153276,
			 70386486105843025439939619828917593665686757934951,
			 62176457141856560629502157223196586755079324193331,
			 64906352462741904929101432445813822663347944758178,
			 92575867718337217661963751590579239728245598838407,
			 58203565325359399008402633568948830189458628227828,
			 80181199384826282014278194139940567587151170094390,
			 35398664372827112653829987240784473053190104293586,
			 86515506006295864861532075273371959191420517255829,
			 71693888707715466499115593487603532921714970056938,
			 54370070576826684624621495650076471787294438377604,
			 53282654108756828443191190634694037855217779295145,
			 36123272525000296071075082563815656710885258350721,
			 45876576172410976447339110607218265236877223636045,
			 17423706905851860660448207621209813287860733969412,
			 81142660418086830619328460811191061556940512689692,
			 51934325451728388641918047049293215058642563049483,
			 62467221648435076201727918039944693004732956340691,
			 15732444386908125794514089057706229429197107928209,
			 55037687525678773091862540744969844508330393682126,
			 18336384825330154686196124348767681297534375946515,
			 80386287592878490201521685554828717201219257766954,
			 78182833757993103614740356856449095527097864797581,
			 16726320100436897842553539920931837441497806860984,
			 48403098129077791799088218795327364475675590848030,
			 87086987551392711854517078544161852424320693150332,
			 59959406895756536782107074926966537676326235447210,
			 69793950679652694742597709739166693763042633987085,
			 41052684708299085211399427365734116182760315001271,
			 65378607361501080857009149939512557028198746004375,
			 35829035317434717326932123578154982629742552737307,
			 94953759765105305946966067683156574377167401875275,
			 88902802571733229619176668713819931811048770190271,
			 25267680276078003013678680992525463401061632866526,
			 36270218540497705585629946580636237993140746255962,
			 24074486908231174977792365466257246923322810917141,
			 91430288197103288597806669760892938638285025333403,
			 34413065578016127815921815005561868836468420090470,
			 23053081172816430487623791969842487255036638784583,
			 11487696932154902810424020138335124462181441773470,
			 63783299490636259666498587618221225225512486764533,
			 67720186971698544312419572409913959008952310058822,
			 95548255300263520781532296796249481641953868218774,
			 76085327132285723110424803456124867697064507995236,
			 37774242535411291684276865538926205024910326572967,
			 23701913275725675285653248258265463092207058596522,
			 29798860272258331913126375147341994889534765745501,
			 18495701454879288984856827726077713721403798879715,
			 38298203783031473527721580348144513491373226651381,
			 34829543829199918180278916522431027392251122869539,
			 40957953066405232632538044100059654939159879593635,
			 29746152185502371307642255121183693803580388584903,
			 41698116222072977186158236678424689157993532961922,
			 62467957194401269043877107275048102390895523597457,
			 23189706772547915061505504953922979530901129967519,
			 86188088225875314529584099251203829009407770775672,
			 11306739708304724483816533873502340845647058077308,
			 82959174767140363198008187129011875491310547126581,
			 97623331044818386269515456334926366572897563400500,
			 42846280183517070527831839425882145521227251250327,
			 55121603546981200581762165212827652751691296897789,
			 32238195734329339946437501907836945765883352399886,
			 75506164965184775180738168837861091527357929701337,
			 62177842752192623401942399639168044983993173312731,
			 32924185707147349566916674687634660915035914677504,
			 99518671430235219628894890102423325116913619626622,
			 73267460800591547471830798392868535206946944540724,
			 76841822524674417161514036427982273348055556214818,
			 97142617910342598647204516893989422179826088076852,
			 87783646182799346313767754307809363333018982642090,
			 10848802521674670883215120185883543223812876952786,
			 71329612474782464538636993009049310363619763878039,
			 62184073572399794223406235393808339651327408011116,
			 66627891981488087797941876876144230030984490851411,
			 60661826293682836764744779239180335110989069790714,
			 85786944089552990653640447425576083659976645795096,
			 66024396409905389607120198219976047599490197230297,
			 64913982680032973156037120041377903785566085089252,
			 16730939319872750275468906903707539413042652315011,
			 94809377245048795150954100921645863754710598436791,
			 78639167021187492431995700641917969777599028300699,
			 15368713711936614952811305876380278410754449733078,
			 40789923115535562561142322423255033685442488917353,
			 44889911501440648020369068063960672322193204149535,
			 41503128880339536053299340368006977710650566631954,
			 81234880673210146739058568557934581403627822703280,
			 82616570773948327592232845941706525094512325230608,
			 22918802058777319719839450180888072429661980811197,
			 77158542502016545090413245809786882778948721859617,
			 72107838435069186155435662884062257473692284509516,
			 20849603980134001723930671666823555245252804609722,
			 53503534226472524250874054075591789781264330331690]
		
	
	sumstr = str(sum(terms))
	return sumstr[:kwargs['numdigits']]

# -----------------------------
# -----------------------------

@SetupAndTime(14,'Starting number, under %(terms) which produces the longest chain:', terms = 1000000)
def problem14(**kwargs):

	def transform(n):
		if n%2 == 0: return n/2
		else:        return 3*n+1
			
	def getlist(value, nextDict):
		tmp = value
		prior = [value]
		while tmp in nextDict:
			prior.append(nextDict[tmp])
			tmp = prior[-1]
			
		return prior
			
	
	numtermstosearch = kwargs['terms']
	
	longest = 1
	
	# Cache existing sequences to prevent tons of repetition.
	nextVal = dict()
	lengthFrom = dict()	
	
	lengthFrom[1] = 0
	
	# Assume the term must be an odd term.  Even terms always get smaller, while odd terms always can get larger.
	# If we know an odd term is up front, then we only need to search all odd terms.
	for i in xrange(1, numtermstosearch+1, 2):
	
		# If we already see the term, skip it and don't recalculate it.			
		if i in lengthFrom: continue
	
		termFrom = i
		sequence = [termFrom]

		while termFrom != 1 and not termFrom in nextVal:
			termTo = transform(termFrom)
			nextVal[termFrom] = termTo
			sequence.append(termTo)
			
			termFrom = termTo
				
		sequence = sequence[::-1]
			
		for term in zip(sequence, range(len(sequence))):
			if term[0] in lengthFrom:
				# Ignore, we've already added this sequence.
				continue
			else:
				# Check to see what the length of the next term is to 1, and add 1 to this.
				lengthFrom[term[0]] = lengthFrom[nextVal[term[0]]] + 1
			
		if lengthFrom[i] > lengthFrom[longest]:
			longest = i
	
	termSequence = getlist(longest, nextVal)
	
	return stringify(longest, ' - length', len(termSequence), termSequence[:5], '...', termSequence[-5:])
	
# -----------------------------
# -----------------------------

@SetupAndTime(15,'Number of lattice paths for a %(maxgrid) x %(maxgrid) grid:', maxgrid = 20)
def problem15(**kwargs):

	maxgrid = kwargs['maxgrid']

	for gridsize in xrange(1,maxgrid+1):
		paths = 1
		for i in xrange(gridsize):
			# Each 1x1 is worth 2 paths, but shares i boarders with the existing grid.
			paths *= (2*gridsize) - i
			
			# The expanded grid shares edges.
			paths /= i+1
			
	return paths
		
# -----------------------------
# -----------------------------

@SetupAndTime(16,'Sum of the digits in %(base)^%(power):', base = 2, power = 1000)
def problem16(**kwargs):
	base, power = kwargs['base'], kwargs['power']
	return sum([int(val) for val in str(pow(base, power))])
		
# -----------------------------
# -----------------------------

@SetupAndTime(17,'Number of letters used in the word representation of the first %(numTerms) numbers:', numTerms=1000)
def problem17(**kwargs):

	def handleHundredTerm(term, uniques, tens):
					
		remainder = int(term)
		hundredTerm = remainder / 100
		
		remainder -= (hundredTerm*100)
		tenTerm = remainder / 10
		
		remainder -= (tenTerm*10)
		oneTerm = remainder
		
		term = ''

		if hundredTerm > 0:
			term = uniques[hundredTerm] + ' hundred '
			
		if len(term) > 0 and (oneTerm > 0 or tenTerm > 0):
			term += 'and '
			
		# Quick hax to handle one through ninteen
		uniquecheck = int(str(tenTerm)+str(oneTerm))
		if uniquecheck < 20:
			term += uniques[uniquecheck]
			
		else:
			# Append the tens term.
			term += tens[tenTerm]
			
			# Append the ones term, if it exists.
			if oneTerm > 0:
				term += '-' + uniques[oneTerm]
			else:
				term += uniques[oneTerm]
				
		return term

	def wordify(number):

		suffixes = ['', 'thousand', 'million', 'billion', 'trillion']
		tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
		uniques = ['','one', 'two', 'three', 'four', 'five','six','seven','eight','nine',
				   'ten','eleven','twelve','thirteen','fourteen', 'fifteen','sixteen','seventeen','eighteen','nineteen']
		
		if number == 0:
			return 'zero'
		if number < 20:
			return uniques[number]
			
		# Partition number into groups of three.
		strnum = str(number)
		separated = [strnum[max(i-3,0):i] for i in range(len(strnum), 0, -3)]
		
		strs = []
		suffixNum = 0
		for part in separated:		
			value = handleHundredTerm(part, uniques, tens).lstrip().rstrip()
			
			if value != '':
				value += ' ' + suffixes[suffixNum]
				value = value.lstrip().rstrip()
				
			strs.append(value)
			suffixNum += 1
		
		strs = strs[::-1]
		return ' '.join(strs)
		
	numTerms = kwargs['numTerms']
	terms = []
	totalsize = 0
		
	for i in xrange(1, numTerms+1):
		terms.append(wordify(i))
		
		tmp = terms[-1]
		tmp = tmp.replace(' ', '')
		tmp = tmp.replace('-', '')
		totalsize += len(tmp)
				
	return totalsize
		
# -----------------------------
# -----------------------------

# !!!! Solves both problems 18 and 67 !!!!
def problemPYRAMID(termString):

	# Match a level to the terms above it.	
	def groupweights(lowerlevel, upperlevel):
	
		upperValues = []
				
		for i in xrange(len(upperlevel)):
			left, right = lowerlevel[i], lowerlevel[i+1]
			newelement = upperlevel[i]
			
			if left[0] >= right[0]:
				newelement[0] += left[0]
				newelement[1] += '->' + left[1]

			else:
				newelement[0] += right[0]
				newelement[1] += '->' + right[1]
				
			upperValues.append(newelement)

		return upperValues
			
					
	def emptyFormatLines(lines):
		return [[line, str(line)] for line in lines]
			
	# Start from the end, work towards the top.
	levels = [[int(val) for val in elem.lstrip().rstrip().split(' ')] for elem in termString][::-1]
	levels = [emptyFormatLines(line) for line in levels]

	
	for level in xrange(0, len(levels)-1):
		levels[level+1] = groupweights(levels[level], levels[level+1])
		
	
	return stringify('Sum:', levels[-1][0][0], '\nTerms:', levels[-1][0][1])
	
@SetupAndTime(18,'Greatest sum of a path going down a small pyramid:')
def problem18(**kwargs):
	string =  ["75                                          ",
			   "95 64                                       ",
			   "17 47 82                                    ",
			   "18 35 87 10                                 ",
			   "20 04 82 47 65                              ",
			   "19 01 23 75 03 34                           ",
			   "88 02 77 73 07 63 67                        ",
			   "99 65 04 28 06 16 70 92                     ",
			   "41 41 26 56 83 40 80 70 33                  ",
			   "41 48 72 33 47 32 37 16 94 29               ",
			   "53 71 44 65 25 43 91 52 97 51 14            ",
			   "70 11 33 28 77 73 17 78 39 68 17 57         ",
			   "91 71 52 38 17 14 91 43 58 50 27 29 48      ",
			   "63 66 04 68 89 53 67 30 73 16 69 87 40 31   ",
			   "04 62 98 27 23 09 70 98 73 93 38 53 60 04 23"]
	
	return problemPYRAMID(string)		
	
@SetupAndTime(67,'Greatest sum of a path going down a large pyramid:')
def problem67(**kwargs):
	string =  ["59",
			   "73 41",
			   "52 40 09",
			   "26 53 06 34",
			   "10 51 87 86 81",
			   "61 95 66 57 25 68",
			   "90 81 80 38 92 67 73",
			   "30 28 51 76 81 18 75 44",
			   "84 14 95 87 62 81 17 78 58",
			   "21 46 71 58 02 79 62 39 31 09",
			   "56 34 35 53 78 31 81 18 90 93 15",
			   "78 53 04 21 84 93 32 13 97 11 37 51",
			   "45 03 81 79 05 18 78 86 13 30 63 99 95",
			   "39 87 96 28 03 38 42 17 82 87 58 07 22 57",
			   "06 17 51 17 07 93 09 07 75 97 95 78 87 08 53",
			   "67 66 59 60 88 99 94 65 55 77 55 34 27 53 78 28",
			   "76 40 41 04 87 16 09 42 75 69 23 97 30 60 10 79 87",
			   "12 10 44 26 21 36 32 84 98 60 13 12 36 16 63 31 91 35",
			   "70 39 06 05 55 27 38 48 28 22 34 35 62 62 15 14 94 89 86",
			   "66 56 68 84 96 21 34 34 34 81 62 40 65 54 62 05 98 03 02 60",
			   "38 89 46 37 99 54 34 53 36 14 70 26 02 90 45 13 31 61 83 73 47",
			   "36 10 63 96 60 49 41 05 37 42 14 58 84 93 96 17 09 43 05 43 06 59",
			   "66 57 87 57 61 28 37 51 84 73 79 15 39 95 88 87 43 39 11 86 77 74 18",
			   "54 42 05 79 30 49 99 73 46 37 50 02 45 09 54 52 27 95 27 65 19 45 26 45",
			   "71 39 17 78 76 29 52 90 18 99 78 19 35 62 71 19 23 65 93 85 49 33 75 09 02",
			   "33 24 47 61 60 55 32 88 57 55 91 54 46 57 07 77 98 52 80 99 24 25 46 78 79 05",
			   "92 09 13 55 10 67 26 78 76 82 63 49 51 31 24 68 05 57 07 54 69 21 67 43 17 63 12",
			   "24 59 06 08 98 74 66 26 61 60 13 03 09 09 24 30 71 08 88 70 72 70 29 90 11 82 41 34",
			   "66 82 67 04 36 60 92 77 91 85 62 49 59 61 30 90 29 94 26 41 89 04 53 22 83 41 09 74 90",
			   "48 28 26 37 28 52 77 26 51 32 18 98 79 36 62 13 17 08 19 54 89 29 73 68 42 14 08 16 70 37",
			   "37 60 69 70 72 71 09 59 13 60 38 13 57 36 09 30 43 89 30 39 15 02 44 73 05 73 26 63 56 86 12",
			   "55 55 85 50 62 99 84 77 28 85 03 21 27 22 19 26 82 69 54 04 13 07 85 14 01 15 70 59 89 95 10 19",
			   "04 09 31 92 91 38 92 86 98 75 21 05 64 42 62 84 36 20 73 42 21 23 22 51 51 79 25 45 85 53 03 43 22",
			   "75 63 02 49 14 12 89 14 60 78 92 16 44 82 38 30 72 11 46 52 90 27 08 65 78 03 85 41 57 79 39 52 33 48",
			   "78 27 56 56 39 13 19 43 86 72 58 95 39 07 04 34 21 98 39 15 39 84 89 69 84 46 37 57 59 35 59 50 26 15 93",
			   "42 89 36 27 78 91 24 11 17 41 05 94 07 69 51 96 03 96 47 90 90 45 91 20 50 56 10 32 36 49 04 53 85 92 25 65",
			   "52 09 61 30 61 97 66 21 96 92 98 90 06 34 96 60 32 69 68 33 75 84 18 31 71 50 84 63 03 03 19 11 28 42 75 45 45",
			   "61 31 61 68 96 34 49 39 05 71 76 59 62 67 06 47 96 99 34 21 32 47 52 07 71 60 42 72 94 56 82 83 84 40 94 87 82 46",
			   "01 20 60 14 17 38 26 78 66 81 45 95 18 51 98 81 48 16 53 88 37 52 69 95 72 93 22 34 98 20 54 27 73 61 56 63 60 34 63",
			   "93 42 94 83 47 61 27 51 79 79 45 01 44 73 31 70 83 42 88 25 53 51 30 15 65 94 80 44 61 84 12 77 02 62 02 65 94 42 14 94",
			   "32 73 09 67 68 29 74 98 10 19 85 48 38 31 85 67 53 93 93 77 47 67 39 72 94 53 18 43 77 40 78 32 29 59 24 06 02 83 50 60 66",
			   "32 01 44 30 16 51 15 81 98 15 10 62 86 79 50 62 45 60 70 38 31 85 65 61 64 06 69 84 14 22 56 43 09 48 66 69 83 91 60 40 36 61",
			   "92 48 22 99 15 95 64 43 01 16 94 02 99 19 17 69 11 58 97 56 89 31 77 45 67 96 12 73 08 20 36 47 81 44 50 64 68 85 40 81 85 52 09",
			   "91 35 92 45 32 84 62 15 19 64 21 66 06 01 52 80 62 59 12 25 88 28 91 50 40 16 22 99 92 79 87 51 21 77 74 77 07 42 38 42 74 83 02 05",
			   "46 19 77 66 24 18 05 32 02 84 31 99 92 58 96 72 91 36 62 99 55 29 53 42 12 37 26 58 89 50 66 19 82 75 12 48 24 87 91 85 02 07 03 76 86",
			   "99 98 84 93 07 17 33 61 92 20 66 60 24 66 40 30 67 05 37 29 24 96 03 27 70 62 13 04 45 47 59 88 43 20 66 15 46 92 30 04 71 66 78 70 53 99",
			   "67 60 38 06 88 04 17 72 10 99 71 07 42 25 54 05 26 64 91 50 45 71 06 30 67 48 69 82 08 56 80 67 18 46 66 63 01 20 08 80 47 07 91 16 03 79 87",
			   "18 54 78 49 80 48 77 40 68 23 60 88 58 80 33 57 11 69 55 53 64 02 94 49 60 92 16 35 81 21 82 96 25 24 96 18 02 05 49 03 50 77 06 32 84 27 18 38",
			   "68 01 50 04 03 21 42 94 53 24 89 05 92 26 52 36 68 11 85 01 04 42 02 45 15 06 50 04 53 73 25 74 81 88 98 21 67 84 79 97 99 20 95 04 40 46 02 58 87",
			   "94 10 02 78 88 52 21 03 88 60 06 53 49 71 20 91 12 65 07 49 21 22 11 41 58 99 36 16 09 48 17 24 52 36 23 15 72 16 84 56 02 99 43 76 81 71 29 39 49 17",
			   "64 39 59 84 86 16 17 66 03 09 43 06 64 18 63 29 68 06 23 07 87 14 26 35 17 12 98 41 53 64 78 18 98 27 28 84 80 67 75 62 10 11 76 90 54 10 05 54 41 39 66",
			   "43 83 18 37 32 31 52 29 95 47 08 76 35 11 04 53 35 43 34 10 52 57 12 36 20 39 40 55 78 44 07 31 38 26 08 15 56 88 86 01 52 62 10 24 32 05 60 65 53 28 57 99",
			   "03 50 03 52 07 73 49 92 66 80 01 46 08 67 25 36 73 93 07 42 25 53 13 96 76 83 87 90 54 89 78 22 78 91 73 51 69 09 79 94 83 53 09 40 69 62 10 79 49 47 03 81 30",
			   "71 54 73 33 51 76 59 54 79 37 56 45 84 17 62 21 98 69 41 95 65 24 39 37 62 03 24 48 54 64 46 82 71 78 33 67 09 16 96 68 52 74 79 68 32 21 13 78 96 60 09 69 20 36",
			   "73 26 21 44 46 38 17 83 65 98 07 23 52 46 61 97 33 13 60 31 70 15 36 77 31 58 56 93 75 68 21 36 69 53 90 75 25 82 39 50 65 94 29 30 11 33 11 13 96 02 56 47 07 49 02",
			   "76 46 73 30 10 20 60 70 14 56 34 26 37 39 48 24 55 76 84 91 39 86 95 61 50 14 53 93 64 67 37 31 10 84 42 70 48 20 10 72 60 61 84 79 69 65 99 73 89 25 85 48 92 56 97 16",
			   "03 14 80 27 22 30 44 27 67 75 79 32 51 54 81 29 65 14 19 04 13 82 04 91 43 40 12 52 29 99 07 76 60 25 01 07 61 71 37 92 40 47 99 66 57 01 43 44 22 40 53 53 09 69 26 81 07",
			   "49 80 56 90 93 87 47 13 75 28 87 23 72 79 32 18 27 20 28 10 37 59 21 18 70 04 79 96 03 31 45 71 81 06 14 18 17 05 31 50 92 79 23 47 09 39 47 91 43 54 69 47 42 95 62 46 32 85",
			   "37 18 62 85 87 28 64 05 77 51 47 26 30 65 05 70 65 75 59 80 42 52 25 20 44 10 92 17 71 95 52 14 77 13 24 55 11 65 26 91 01 30 63 15 49 48 41 17 67 47 03 68 20 90 98 32 04 40 68",
			   "90 51 58 60 06 55 23 68 05 19 76 94 82 36 96 43 38 90 87 28 33 83 05 17 70 83 96 93 06 04 78 47 80 06 23 84 75 23 87 72 99 14 50 98 92 38 90 64 61 58 76 94 36 66 87 80 51 35 61 38",
			   "57 95 64 06 53 36 82 51 40 33 47 14 07 98 78 65 39 58 53 06 50 53 04 69 40 68 36 69 75 78 75 60 03 32 39 24 74 47 26 90 13 40 44 71 90 76 51 24 36 50 25 45 70 80 61 80 61 43 90 64 11",
			   "18 29 86 56 68 42 79 10 42 44 30 12 96 18 23 18 52 59 02 99 67 46 60 86 43 38 55 17 44 93 42 21 55 14 47 34 55 16 49 24 23 29 96 51 55 10 46 53 27 92 27 46 63 57 30 65 43 27 21 20 24 83",
			   "81 72 93 19 69 52 48 01 13 83 92 69 20 48 69 59 20 62 05 42 28 89 90 99 32 72 84 17 08 87 36 03 60 31 36 36 81 26 97 36 48 54 56 56 27 16 91 08 23 11 87 99 33 47 02 14 44 73 70 99 43 35 33",
			   "90 56 61 86 56 12 70 59 63 32 01 15 81 47 71 76 95 32 65 80 54 70 34 51 40 45 33 04 64 55 78 68 88 47 31 47 68 87 03 84 23 44 89 72 35 08 31 76 63 26 90 85 96 67 65 91 19 14 17 86 04 71 32 95",
			   "37 13 04 22 64 37 37 28 56 62 86 33 07 37 10 44 52 82 52 06 19 52 57 75 90 26 91 24 06 21 14 67 76 30 46 14 35 89 89 41 03 64 56 97 87 63 22 34 03 79 17 45 11 53 25 56 96 61 23 18 63 31 37 37 47",
			   "77 23 26 70 72 76 77 04 28 64 71 69 14 85 96 54 95 48 06 62 99 83 86 77 97 75 71 66 30 19 57 90 33 01 60 61 14 12 90 99 32 77 56 41 18 14 87 49 10 14 90 64 18 50 21 74 14 16 88 05 45 73 82 47 74 44",
			   "22 97 41 13 34 31 54 61 56 94 03 24 59 27 98 77 04 09 37 40 12 26 87 09 71 70 07 18 64 57 80 21 12 71 83 94 60 39 73 79 73 19 97 32 64 29 41 07 48 84 85 67 12 74 95 20 24 52 41 67 56 61 29 93 35 72 69",
			   "72 23 63 66 01 11 07 30 52 56 95 16 65 26 83 90 50 74 60 18 16 48 43 77 37 11 99 98 30 94 91 26 62 73 45 12 87 73 47 27 01 88 66 99 21 41 95 80 02 53 23 32 61 48 32 43 43 83 14 66 95 91 19 81 80 67 25 88",
			   "08 62 32 18 92 14 83 71 37 96 11 83 39 99 05 16 23 27 10 67 02 25 44 11 55 31 46 64 41 56 44 74 26 81 51 31 45 85 87 09 81 95 22 28 76 69 46 48 64 87 67 76 27 89 31 11 74 16 62 03 60 94 42 47 09 34 94 93 72",
			   "56 18 90 18 42 17 42 32 14 86 06 53 33 95 99 35 29 15 44 20 49 59 25 54 34 59 84 21 23 54 35 90 78 16 93 13 37 88 54 19 86 67 68 55 66 84 65 42 98 37 87 56 33 28 58 38 28 38 66 27 52 21 81 15 08 22 97 32 85 27",
			   "91 53 40 28 13 34 91 25 01 63 50 37 22 49 71 58 32 28 30 18 68 94 23 83 63 62 94 76 80 41 90 22 82 52 29 12 18 56 10 08 35 14 37 57 23 65 67 40 72 39 93 39 70 89 40 34 07 46 94 22 20 05 53 64 56 30 05 56 61 88 27",
			   "23 95 11 12 37 69 68 24 66 10 87 70 43 50 75 07 62 41 83 58 95 93 89 79 45 39 02 22 05 22 95 43 62 11 68 29 17 40 26 44 25 71 87 16 70 85 19 25 59 94 90 41 41 80 61 70 55 60 84 33 95 76 42 63 15 09 03 40 38 12 03 32",
			   "09 84 56 80 61 55 85 97 16 94 82 94 98 57 84 30 84 48 93 90 71 05 95 90 73 17 30 98 40 64 65 89 07 79 09 19 56 36 42 30 23 69 73 72 07 05 27 61 24 31 43 48 71 84 21 28 26 65 65 59 65 74 77 20 10 81 61 84 95 08 52 23 70",
			   "47 81 28 09 98 51 67 64 35 51 59 36 92 82 77 65 80 24 72 53 22 07 27 10 21 28 30 22 48 82 80 48 56 20 14 43 18 25 50 95 90 31 77 08 09 48 44 80 90 22 93 45 82 17 13 96 25 26 08 73 34 99 06 49 24 06 83 51 40 14 15 10 25 01",
			   "54 25 10 81 30 64 24 74 75 80 36 75 82 60 22 69 72 91 45 67 03 62 79 54 89 74 44 83 64 96 66 73 44 30 74 50 37 05 09 97 70 01 60 46 37 91 39 75 75 18 58 52 72 78 51 81 86 52 08 97 01 46 43 66 98 62 81 18 70 93 73 08 32 46 34",
			   "96 80 82 07 59 71 92 53 19 20 88 66 03 26 26 10 24 27 50 82 94 73 63 08 51 33 22 45 19 13 58 33 90 15 22 50 36 13 55 06 35 47 82 52 33 61 36 27 28 46 98 14 73 20 73 32 16 26 80 53 47 66 76 38 94 45 02 01 22 52 47 96 64 58 52 39",
			   "88 46 23 39 74 63 81 64 20 90 33 33 76 55 58 26 10 46 42 26 74 74 12 83 32 43 09 02 73 55 86 54 85 34 28 23 29 79 91 62 47 41 82 87 99 22 48 90 20 05 96 75 95 04 43 28 81 39 81 01 28 42 78 25 39 77 90 57 58 98 17 36 73 22 63 74 51",
			   "29 39 74 94 95 78 64 24 38 86 63 87 93 06 70 92 22 16 80 64 29 52 20 27 23 50 14 13 87 15 72 96 81 22 08 49 72 30 70 24 79 31 16 64 59 21 89 34 96 91 48 76 43 53 88 01 57 80 23 81 90 79 58 01 80 87 17 99 86 90 72 63 32 69 14 28 88 69",
			   "37 17 71 95 56 93 71 35 43 45 04 98 92 94 84 96 11 30 31 27 31 60 92 03 48 05 98 91 86 94 35 90 90 08 48 19 33 28 68 37 59 26 65 96 50 68 22 07 09 49 34 31 77 49 43 06 75 17 81 87 61 79 52 26 27 72 29 50 07 98 86 01 17 10 46 64 24 18 56",
			   "51 30 25 94 88 85 79 91 40 33 63 84 49 67 98 92 15 26 75 19 82 05 18 78 65 93 61 48 91 43 59 41 70 51 22 15 92 81 67 91 46 98 11 11 65 31 66 10 98 65 83 21 05 56 05 98 73 67 46 74 69 34 08 30 05 52 07 98 32 95 30 94 65 50 24 63 28 81 99 57",
			   "19 23 61 36 09 89 71 98 65 17 30 29 89 26 79 74 94 11 44 48 97 54 81 55 39 66 69 45 28 47 13 86 15 76 74 70 84 32 36 33 79 20 78 14 41 47 89 28 81 05 99 66 81 86 38 26 06 25 13 60 54 55 23 53 27 05 89 25 23 11 13 54 59 54 56 34 16 24 53 44 06",
			   "13 40 57 72 21 15 60 08 04 19 11 98 34 45 09 97 86 71 03 15 56 19 15 44 97 31 90 04 87 87 76 08 12 30 24 62 84 28 12 85 82 53 99 52 13 94 06 65 97 86 09 50 94 68 69 74 30 67 87 94 63 07 78 27 80 36 69 41 06 92 32 78 37 82 30 05 18 87 99 72 19 99",
			   "44 20 55 77 69 91 27 31 28 81 80 27 02 07 97 23 95 98 12 25 75 29 47 71 07 47 78 39 41 59 27 76 13 15 66 61 68 35 69 86 16 53 67 63 99 85 41 56 08 28 33 40 94 76 90 85 31 70 24 65 84 65 99 82 19 25 54 37 21 46 33 02 52 99 51 33 26 04 87 02 08 18 96",
			   "54 42 61 45 91 06 64 79 80 82 32 16 83 63 42 49 19 78 65 97 40 42 14 61 49 34 04 18 25 98 59 30 82 72 26 88 54 36 21 75 03 88 99 53 46 51 55 78 22 94 34 40 68 87 84 25 30 76 25 08 92 84 42 61 40 38 09 99 40 23 29 39 46 55 10 90 35 84 56 70 63 23 91 39",
			   "52 92 03 71 89 07 09 37 68 66 58 20 44 92 51 56 13 71 79 99 26 37 02 06 16 67 36 52 58 16 79 73 56 60 59 27 44 77 94 82 20 50 98 33 09 87 94 37 40 83 64 83 58 85 17 76 53 02 83 52 22 27 39 20 48 92 45 21 09 42 24 23 12 37 52 28 50 78 79 20 86 62 73 20 59",
			   "54 96 80 15 91 90 99 70 10 09 58 90 93 50 81 99 54 38 36 10 30 11 35 84 16 45 82 18 11 97 36 43 96 79 97 65 40 48 23 19 17 31 64 52 65 65 37 32 65 76 99 79 34 65 79 27 55 33 03 01 33 27 61 28 66 08 04 70 49 46 48 83 01 45 19 96 13 81 14 21 31 79 93 85 50 05",
			   "92 92 48 84 59 98 31 53 23 27 15 22 79 95 24 76 05 79 16 93 97 89 38 89 42 83 02 88 94 95 82 21 01 97 48 39 31 78 09 65 50 56 97 61 01 07 65 27 21 23 14 15 80 97 44 78 49 35 33 45 81 74 34 05 31 57 09 38 94 07 69 54 69 32 65 68 46 68 78 90 24 28 49 51 45 86 35",
			   "41 63 89 76 87 31 86 09 46 14 87 82 22 29 47 16 13 10 70 72 82 95 48 64 58 43 13 75 42 69 21 12 67 13 64 85 58 23 98 09 37 76 05 22 31 12 66 50 29 99 86 72 45 25 10 28 19 06 90 43 29 31 67 79 46 25 74 14 97 35 76 37 65 46 23 82 06 22 30 76 93 66 94 17 96 13 20 72",
			   "63 40 78 08 52 09 90 41 70 28 36 14 46 44 85 96 24 52 58 15 87 37 05 98 99 39 13 61 76 38 44 99 83 74 90 22 53 80 56 98 30 51 63 39 44 30 91 91 04 22 27 73 17 35 53 18 35 45 54 56 27 78 48 13 69 36 44 38 71 25 30 56 15 22 73 43 32 69 59 25 93 83 45 11 34 94 44 39 92",
			   "12 36 56 88 13 96 16 12 55 54 11 47 19 78 17 17 68 81 77 51 42 55 99 85 66 27 81 79 93 42 65 61 69 74 14 01 18 56 12 01 58 37 91 22 42 66 83 25 19 04 96 41 25 45 18 69 96 88 36 93 10 12 98 32 44 83 83 04 72 91 04 27 73 07 34 37 71 60 59 31 01 54 54 44 96 93 83 36 04 45",
			   "30 18 22 20 42 96 65 79 17 41 55 69 94 81 29 80 91 31 85 25 47 26 43 49 02 99 34 67 99 76 16 14 15 93 08 32 99 44 61 77 67 50 43 55 87 55 53 72 17 46 62 25 50 99 73 05 93 48 17 31 70 80 59 09 44 59 45 13 74 66 58 94 87 73 16 14 85 38 74 99 64 23 79 28 71 42 20 37 82 31 23",
			   "51 96 39 65 46 71 56 13 29 68 53 86 45 33 51 49 12 91 21 21 76 85 02 17 98 15 46 12 60 21 88 30 92 83 44 59 42 50 27 88 46 86 94 73 45 54 23 24 14 10 94 21 20 34 23 51 04 83 99 75 90 63 60 16 22 33 83 70 11 32 10 50 29 30 83 46 11 05 31 17 86 42 49 01 44 63 28 60 07 78 95 40",
			   "44 61 89 59 04 49 51 27 69 71 46 76 44 04 09 34 56 39 15 06 94 91 75 90 65 27 56 23 74 06 23 33 36 69 14 39 05 34 35 57 33 22 76 46 56 10 61 65 98 09 16 69 04 62 65 18 99 76 49 18 72 66 73 83 82 40 76 31 89 91 27 88 17 35 41 35 32 51 32 67 52 68 74 85 80 57 07 11 62 66 47 22 67",
			   "65 37 19 97 26 17 16 24 24 17 50 37 64 82 24 36 32 11 68 34 69 31 32 89 79 93 96 68 49 90 14 23 04 04 67 99 81 74 70 74 36 96 68 09 64 39 88 35 54 89 96 58 66 27 88 97 32 14 06 35 78 20 71 06 85 66 57 02 58 91 72 05 29 56 73 48 86 52 09 93 22 57 79 42 12 01 31 68 17 59 63 76 07 77",
			   "73 81 14 13 17 20 11 09 01 83 08 85 91 70 84 63 62 77 37 07 47 01 59 95 39 69 39 21 99 09 87 02 97 16 92 36 74 71 90 66 33 73 73 75 52 91 11 12 26 53 05 26 26 48 61 50 90 65 01 87 42 47 74 35 22 73 24 26 56 70 52 05 48 41 31 18 83 27 21 39 80 85 26 08 44 02 71 07 63 22 05 52 19 08 20",
			   "17 25 21 11 72 93 33 49 64 23 53 82 03 13 91 65 85 02 40 05 42 31 77 42 05 36 06 54 04 58 07 76 87 83 25 57 66 12 74 33 85 37 74 32 20 69 03 97 91 68 82 44 19 14 89 28 85 85 80 53 34 87 58 98 88 78 48 65 98 40 11 57 10 67 70 81 60 79 74 72 97 59 79 47 30 20 54 80 89 91 14 05 33 36 79 39",
			   "60 85 59 39 60 07 57 76 77 92 06 35 15 72 23 41 45 52 95 18 64 79 86 53 56 31 69 11 91 31 84 50 44 82 22 81 41 40 30 42 30 91 48 94 74 76 64 58 74 25 96 57 14 19 03 99 28 83 15 75 99 01 89 85 79 50 03 95 32 67 44 08 07 41 62 64 29 20 14 76 26 55 48 71 69 66 19 72 44 25 14 01 48 74 12 98 07",
			   "64 66 84 24 18 16 27 48 20 14 47 69 30 86 48 40 23 16 61 21 51 50 26 47 35 33 91 28 78 64 43 68 04 79 51 08 19 60 52 95 06 68 46 86 35 97 27 58 04 65 30 58 99 12 12 75 91 39 50 31 42 64 70 04 46 07 98 73 98 93 37 89 77 91 64 71 64 65 66 21 78 62 81 74 42 20 83 70 73 95 78 45 92 27 34 53 71 15",
			   "30 11 85 31 34 71 13 48 05 14 44 03 19 67 23 73 19 57 06 90 94 72 57 69 81 62 59 68 88 57 55 69 49 13 07 87 97 80 89 05 71 05 05 26 38 40 16 62 45 99 18 38 98 24 21 26 62 74 69 04 85 57 77 35 58 67 91 79 79 57 86 28 66 34 72 51 76 78 36 95 63 90 08 78 47 63 45 31 22 70 52 48 79 94 15 77 61 67 68",
			   "23 33 44 81 80 92 93 75 94 88 23 61 39 76 22 03 28 94 32 06 49 65 41 34 18 23 08 47 62 60 03 63 33 13 80 52 31 54 73 43 70 26 16 69 57 87 83 31 03 93 70 81 47 95 77 44 29 68 39 51 56 59 63 07 25 70 07 77 43 53 64 03 94 42 95 39 18 01 66 21 16 97 20 50 90 16 70 10 95 69 29 06 25 61 41 26 15 59 63 35"]
	
	return problemPYRAMID(string)		

# -----------------------------
# -----------------------------

@SetupAndTime(19,'Sundays between %(start) and (%end):', start = '01/01/1901', end = '12/31/2000')
def problem19(**kwargs):

	# Hax it up with time libraries, because it's easier to let the computer do
	# the math than me do it.  Plus it's perfectly allowable, I would presume.
	
	startDate = kwargs['start']
	endDate = kwargs['end']
	
	dayOn = datetime.datetime.strptime(startDate, '%m/%d/%Y')
	endDay = datetime.datetime.strptime(endDate, '%m/%d/%Y')
		
	firstSunday = 0
	
	while dayOn <= endDay:
	
		dayDate = [int(term) for term in dayOn.strftime('%w,%d').split(',')]
		if dayDate[0] == 0 and dayDate[1] == 1:
			firstSunday += 1
	
		dayOn += datetime.timedelta(days=1)
	
	return firstSunday
		
# -----------------------------
# -----------------------------

@SetupAndTime(20,'Factorial digit sum of %(term)! :', term = 100)
def problem20(**kwargs):
	digits = [int(digit) for digit in str(math.factorial(kwargs['term']))]
	return sum(digits)

# -----------------------------
# -----------------------------

@SetupAndTime(21,'Sum of amicable numbers below %(term):', term = 10000)
def problem21(**kwargs):

	def divisorSum(n):
		# "...numbers less than n which divide evenly into n"
		# So we need to strip out N from the equation even though 1 is included.
		return sum(Prime.factors(n)) + 1
	
	maxterm = kwargs['term']

	values = set()
	total = 0
	
	for a in xrange(1, maxterm+1):
		
		# Ignore primes because divisorSum(prime) == divisorSum(1), both are empty sets and invalid.
		if Prime.isPrime(a): continue

		# D(a) == b
		b = divisorSum(a)
		
		# A != B
		if a == b: continue
		
		# If b is a prime, we know it won't match to A as well.
		if Prime.isPrime(b): continue
		
		# So does D(b) == a ?
		if divisorSum(b) == a:		
			if not a in values: total += a
			if not b in values: total += b
			values.update((a, b))
		
	return total

# -----------------------------
# -----------------------------

@SetupAndTime(22,'Score for all names in the list:')
def problem22(**kwargs):
	
	def namevalue(name, position):
		return sum([ord(char) - ord('A') + 1 for char in name])*(position+1)
	
	#Thankfully ASCII...
	names = ["MARY","PATRICIA","LINDA","BARBARA","ELIZABETH","JENNIFER","MARIA","SUSAN","MARGARET","DOROTHY","LISA","NANCY","KAREN","BETTY","HELEN","SANDRA","DONNA","CAROL","RUTH","SHARON","MICHELLE","LAURA","SARAH","KIMBERLY","DEBORAH","JESSICA","SHIRLEY","CYNTHIA","ANGELA","MELISSA","BRENDA","AMY","ANNA","REBECCA","VIRGINIA","KATHLEEN","PAMELA","MARTHA","DEBRA","AMANDA","STEPHANIE","CAROLYN","CHRISTINE","MARIE","JANET","CATHERINE","FRANCES","ANN","JOYCE","DIANE","ALICE","JULIE","HEATHER","TERESA","DORIS","GLORIA","EVELYN","JEAN","CHERYL","MILDRED","KATHERINE","JOAN","ASHLEY","JUDITH","ROSE","JANICE","KELLY","NICOLE","JUDY","CHRISTINA","KATHY","THERESA","BEVERLY","DENISE","TAMMY","IRENE","JANE","LORI","RACHEL","MARILYN","ANDREA","KATHRYN","LOUISE","SARA","ANNE","JACQUELINE","WANDA","BONNIE","JULIA","RUBY","LOIS","TINA","PHYLLIS","NORMA","PAULA","DIANA","ANNIE","LILLIAN","EMILY","ROBIN","PEGGY","CRYSTAL","GLADYS","RITA","DAWN","CONNIE","FLORENCE","TRACY","EDNA","TIFFANY","CARMEN","ROSA","CINDY","GRACE","WENDY","VICTORIA","EDITH","KIM","SHERRY","SYLVIA","JOSEPHINE","THELMA","SHANNON","SHEILA","ETHEL","ELLEN","ELAINE","MARJORIE","CARRIE","CHARLOTTE","MONICA","ESTHER","PAULINE","EMMA","JUANITA","ANITA","RHONDA","HAZEL","AMBER","EVA","DEBBIE","APRIL","LESLIE","CLARA","LUCILLE","JAMIE","JOANNE","ELEANOR","VALERIE","DANIELLE","MEGAN","ALICIA","SUZANNE","MICHELE","GAIL","BERTHA","DARLENE","VERONICA","JILL","ERIN","GERALDINE","LAUREN","CATHY","JOANN","LORRAINE","LYNN","SALLY","REGINA","ERICA","BEATRICE","DOLORES","BERNICE","AUDREY","YVONNE","ANNETTE","JUNE","SAMANTHA","MARION","DANA","STACY","ANA","RENEE","IDA","VIVIAN","ROBERTA","HOLLY","BRITTANY","MELANIE","LORETTA","YOLANDA","JEANETTE","LAURIE","KATIE","KRISTEN","VANESSA","ALMA","SUE","ELSIE","BETH","JEANNE","VICKI","CARLA","TARA","ROSEMARY","EILEEN","TERRI","GERTRUDE","LUCY","TONYA","ELLA","STACEY","WILMA","GINA","KRISTIN","JESSIE","NATALIE","AGNES","VERA","WILLIE","CHARLENE","BESSIE","DELORES","MELINDA","PEARL","ARLENE","MAUREEN","COLLEEN","ALLISON","TAMARA","JOY","GEORGIA","CONSTANCE","LILLIE","CLAUDIA","JACKIE","MARCIA","TANYA","NELLIE","MINNIE","MARLENE","HEIDI","GLENDA","LYDIA","VIOLA","COURTNEY","MARIAN","STELLA","CAROLINE","DORA","JO","VICKIE","MATTIE","TERRY","MAXINE","IRMA","MABEL","MARSHA","MYRTLE","LENA","CHRISTY","DEANNA","PATSY","HILDA","GWENDOLYN","JENNIE","NORA","MARGIE","NINA","CASSANDRA","LEAH","PENNY","KAY","PRISCILLA","NAOMI","CAROLE","BRANDY","OLGA","BILLIE","DIANNE","TRACEY","LEONA","JENNY","FELICIA","SONIA","MIRIAM","VELMA","BECKY","BOBBIE","VIOLET","KRISTINA","TONI","MISTY","MAE","SHELLY","DAISY","RAMONA","SHERRI","ERIKA","KATRINA","CLAIRE","LINDSEY","LINDSAY","GENEVA","GUADALUPE","BELINDA","MARGARITA","SHERYL","CORA","FAYE","ADA","NATASHA","SABRINA","ISABEL","MARGUERITE","HATTIE","HARRIET","MOLLY","CECILIA","KRISTI","BRANDI","BLANCHE","SANDY","ROSIE","JOANNA","IRIS","EUNICE","ANGIE","INEZ","LYNDA","MADELINE","AMELIA","ALBERTA","GENEVIEVE","MONIQUE","JODI","JANIE","MAGGIE","KAYLA","SONYA","JAN","LEE","KRISTINE","CANDACE","FANNIE","MARYANN","OPAL","ALISON","YVETTE","MELODY","LUZ","SUSIE","OLIVIA","FLORA","SHELLEY","KRISTY","MAMIE","LULA","LOLA","VERNA","BEULAH","ANTOINETTE","CANDICE","JUANA","JEANNETTE","PAM","KELLI","HANNAH","WHITNEY","BRIDGET","KARLA","CELIA","LATOYA","PATTY","SHELIA","GAYLE","DELLA","VICKY","LYNNE","SHERI","MARIANNE","KARA","JACQUELYN","ERMA","BLANCA","MYRA","LETICIA","PAT","KRISTA","ROXANNE","ANGELICA","JOHNNIE","ROBYN","FRANCIS","ADRIENNE","ROSALIE","ALEXANDRA","BROOKE","BETHANY","SADIE","BERNADETTE","TRACI","JODY","KENDRA","JASMINE","NICHOLE","RACHAEL","CHELSEA","MABLE","ERNESTINE","MURIEL","MARCELLA","ELENA","KRYSTAL","ANGELINA","NADINE","KARI","ESTELLE","DIANNA","PAULETTE","LORA","MONA","DOREEN","ROSEMARIE","ANGEL","DESIREE","ANTONIA","HOPE","GINGER","JANIS","BETSY","CHRISTIE","FREDA","MERCEDES","MEREDITH","LYNETTE","TERI","CRISTINA","EULA","LEIGH","MEGHAN","SOPHIA","ELOISE","ROCHELLE","GRETCHEN","CECELIA","RAQUEL","HENRIETTA","ALYSSA","JANA","KELLEY","GWEN","KERRY","JENNA","TRICIA","LAVERNE","OLIVE","ALEXIS","TASHA","SILVIA","ELVIRA","CASEY","DELIA","SOPHIE","KATE","PATTI","LORENA","KELLIE","SONJA","LILA","LANA","DARLA","MAY","MINDY","ESSIE","MANDY","LORENE","ELSA","JOSEFINA","JEANNIE","MIRANDA","DIXIE","LUCIA","MARTA","FAITH","LELA","JOHANNA","SHARI","CAMILLE","TAMI","SHAWNA","ELISA","EBONY","MELBA","ORA","NETTIE","TABITHA","OLLIE","JAIME","WINIFRED","KRISTIE","MARINA","ALISHA","AIMEE","RENA","MYRNA","MARLA","TAMMIE","LATASHA","BONITA","PATRICE","RONDA","SHERRIE","ADDIE","FRANCINE","DELORIS","STACIE","ADRIANA","CHERI","SHELBY","ABIGAIL","CELESTE","JEWEL","CARA","ADELE","REBEKAH","LUCINDA","DORTHY","CHRIS","EFFIE","TRINA","REBA","SHAWN","SALLIE","AURORA","LENORA","ETTA","LOTTIE","KERRI","TRISHA","NIKKI","ESTELLA","FRANCISCA","JOSIE","TRACIE","MARISSA","KARIN","BRITTNEY","JANELLE","LOURDES","LAUREL","HELENE","FERN","ELVA","CORINNE","KELSEY","INA","BETTIE","ELISABETH","AIDA","CAITLIN","INGRID","IVA","EUGENIA","CHRISTA","GOLDIE","CASSIE","MAUDE","JENIFER","THERESE","FRANKIE","DENA","LORNA","JANETTE","LATONYA","CANDY","MORGAN","CONSUELO","TAMIKA","ROSETTA","DEBORA","CHERIE","POLLY","DINA","JEWELL","FAY","JILLIAN","DOROTHEA","NELL","TRUDY","ESPERANZA","PATRICA","KIMBERLEY","SHANNA","HELENA","CAROLINA","CLEO","STEFANIE","ROSARIO","OLA","JANINE","MOLLIE","LUPE","ALISA","LOU","MARIBEL","SUSANNE","BETTE","SUSANA","ELISE","CECILE","ISABELLE","LESLEY","JOCELYN","PAIGE","JONI","RACHELLE","LEOLA","DAPHNE","ALTA","ESTER","PETRA","GRACIELA","IMOGENE","JOLENE","KEISHA","LACEY","GLENNA","GABRIELA","KERI","URSULA","LIZZIE","KIRSTEN","SHANA","ADELINE","MAYRA","JAYNE","JACLYN","GRACIE","SONDRA","CARMELA","MARISA","ROSALIND","CHARITY","TONIA","BEATRIZ","MARISOL","CLARICE","JEANINE","SHEENA","ANGELINE","FRIEDA","LILY","ROBBIE","SHAUNA","MILLIE","CLAUDETTE","CATHLEEN","ANGELIA","GABRIELLE","AUTUMN","KATHARINE","SUMMER","JODIE","STACI","LEA","CHRISTI","JIMMIE","JUSTINE","ELMA","LUELLA","MARGRET","DOMINIQUE","SOCORRO","RENE","MARTINA","MARGO","MAVIS","CALLIE","BOBBI","MARITZA","LUCILE","LEANNE","JEANNINE","DEANA","AILEEN","LORIE","LADONNA","WILLA","MANUELA","GALE","SELMA","DOLLY","SYBIL","ABBY","LARA","DALE","IVY","DEE","WINNIE","MARCY","LUISA","JERI","MAGDALENA","OFELIA","MEAGAN","AUDRA","MATILDA","LEILA","CORNELIA","BIANCA","SIMONE","BETTYE","RANDI","VIRGIE","LATISHA","BARBRA","GEORGINA","ELIZA","LEANN","BRIDGETTE","RHODA","HALEY","ADELA","NOLA","BERNADINE","FLOSSIE","ILA","GRETA","RUTHIE","NELDA","MINERVA","LILLY","TERRIE","LETHA","HILARY","ESTELA","VALARIE","BRIANNA","ROSALYN","EARLINE","CATALINA","AVA","MIA","CLARISSA","LIDIA","CORRINE","ALEXANDRIA","CONCEPCION","TIA","SHARRON","RAE","DONA","ERICKA","JAMI","ELNORA","CHANDRA","LENORE","NEVA","MARYLOU","MELISA","TABATHA","SERENA","AVIS","ALLIE","SOFIA","JEANIE","ODESSA","NANNIE","HARRIETT","LORAINE","PENELOPE","MILAGROS","EMILIA","BENITA","ALLYSON","ASHLEE","TANIA","TOMMIE","ESMERALDA","KARINA","EVE","PEARLIE","ZELMA","MALINDA","NOREEN","TAMEKA","SAUNDRA","HILLARY","AMIE","ALTHEA","ROSALINDA","JORDAN","LILIA","ALANA","GAY","CLARE","ALEJANDRA","ELINOR","MICHAEL","LORRIE","JERRI","DARCY","EARNESTINE","CARMELLA","TAYLOR","NOEMI","MARCIE","LIZA","ANNABELLE","LOUISA","EARLENE","MALLORY","CARLENE","NITA","SELENA","TANISHA","KATY","JULIANNE","JOHN","LAKISHA","EDWINA","MARICELA","MARGERY","KENYA","DOLLIE","ROXIE","ROSLYN","KATHRINE","NANETTE","CHARMAINE","LAVONNE","ILENE","KRIS","TAMMI","SUZETTE","CORINE","KAYE","JERRY","MERLE","CHRYSTAL","LINA","DEANNE","LILIAN","JULIANA","ALINE","LUANN","KASEY","MARYANNE","EVANGELINE","COLETTE","MELVA","LAWANDA","YESENIA","NADIA","MADGE","KATHIE","EDDIE","OPHELIA","VALERIA","NONA","MITZI","MARI","GEORGETTE","CLAUDINE","FRAN","ALISSA","ROSEANN","LAKEISHA","SUSANNA","REVA","DEIDRE","CHASITY","SHEREE","CARLY","JAMES","ELVIA","ALYCE","DEIRDRE","GENA","BRIANA","ARACELI","KATELYN","ROSANNE","WENDI","TESSA","BERTA","MARVA","IMELDA","MARIETTA","MARCI","LEONOR","ARLINE","SASHA","MADELYN","JANNA","JULIETTE","DEENA","AURELIA","JOSEFA","AUGUSTA","LILIANA","YOUNG","CHRISTIAN","LESSIE","AMALIA","SAVANNAH","ANASTASIA","VILMA","NATALIA","ROSELLA","LYNNETTE","CORINA","ALFREDA","LEANNA","CAREY","AMPARO","COLEEN","TAMRA","AISHA","WILDA","KARYN","CHERRY","QUEEN","MAURA","MAI","EVANGELINA","ROSANNA","HALLIE","ERNA","ENID","MARIANA","LACY","JULIET","JACKLYN","FREIDA","MADELEINE","MARA","HESTER","CATHRYN","LELIA","CASANDRA","BRIDGETT","ANGELITA","JANNIE","DIONNE","ANNMARIE","KATINA","BERYL","PHOEBE","MILLICENT","KATHERYN","DIANN","CARISSA","MARYELLEN","LIZ","LAURI","HELGA","GILDA","ADRIAN","RHEA","MARQUITA","HOLLIE","TISHA","TAMERA","ANGELIQUE","FRANCESCA","BRITNEY","KAITLIN","LOLITA","FLORINE","ROWENA","REYNA","TWILA","FANNY","JANELL","INES","CONCETTA","BERTIE","ALBA","BRIGITTE","ALYSON","VONDA","PANSY","ELBA","NOELLE","LETITIA","KITTY","DEANN","BRANDIE","LOUELLA","LETA","FELECIA","SHARLENE","LESA","BEVERLEY","ROBERT","ISABELLA","HERMINIA","TERRA","CELINA","TORI","OCTAVIA","JADE","DENICE","GERMAINE","SIERRA","MICHELL","CORTNEY","NELLY","DORETHA","SYDNEY","DEIDRA","MONIKA","LASHONDA","JUDI","CHELSEY","ANTIONETTE","MARGOT","BOBBY","ADELAIDE","NAN","LEEANN","ELISHA","DESSIE","LIBBY","KATHI","GAYLA","LATANYA","MINA","MELLISA","KIMBERLEE","JASMIN","RENAE","ZELDA","ELDA","MA","JUSTINA","GUSSIE","EMILIE","CAMILLA","ABBIE","ROCIO","KAITLYN","JESSE","EDYTHE","ASHLEIGH","SELINA","LAKESHA","GERI","ALLENE","PAMALA","MICHAELA","DAYNA","CARYN","ROSALIA","SUN","JACQULINE","REBECA","MARYBETH","KRYSTLE","IOLA","DOTTIE","BENNIE","BELLE","AUBREY","GRISELDA","ERNESTINA","ELIDA","ADRIANNE","DEMETRIA","DELMA","CHONG","JAQUELINE","DESTINY","ARLEEN","VIRGINA","RETHA","FATIMA","TILLIE","ELEANORE","CARI","TREVA","BIRDIE","WILHELMINA","ROSALEE","MAURINE","LATRICE","YONG","JENA","TARYN","ELIA","DEBBY","MAUDIE","JEANNA","DELILAH","CATRINA","SHONDA","HORTENCIA","THEODORA","TERESITA","ROBBIN","DANETTE","MARYJANE","FREDDIE","DELPHINE","BRIANNE","NILDA","DANNA","CINDI","BESS","IONA","HANNA","ARIEL","WINONA","VIDA","ROSITA","MARIANNA","WILLIAM","RACHEAL","GUILLERMINA","ELOISA","CELESTINE","CAREN","MALISSA","LONA","CHANTEL","SHELLIE","MARISELA","LEORA","AGATHA","SOLEDAD","MIGDALIA","IVETTE","CHRISTEN","ATHENA","JANEL","CHLOE","VEDA","PATTIE","TESSIE","TERA","MARILYNN","LUCRETIA","KARRIE","DINAH","DANIELA","ALECIA","ADELINA","VERNICE","SHIELA","PORTIA","MERRY","LASHAWN","DEVON","DARA","TAWANA","OMA","VERDA","CHRISTIN","ALENE","ZELLA","SANDI","RAFAELA","MAYA","KIRA","CANDIDA","ALVINA","SUZAN","SHAYLA","LYN","LETTIE","ALVA","SAMATHA","ORALIA","MATILDE","MADONNA","LARISSA","VESTA","RENITA","INDIA","DELOIS","SHANDA","PHILLIS","LORRI","ERLINDA","CRUZ","CATHRINE","BARB","ZOE","ISABELL","IONE","GISELA","CHARLIE","VALENCIA","ROXANNA","MAYME","KISHA","ELLIE","MELLISSA","DORRIS","DALIA","BELLA","ANNETTA","ZOILA","RETA","REINA","LAURETTA","KYLIE","CHRISTAL","PILAR","CHARLA","ELISSA","TIFFANI","TANA","PAULINA","LEOTA","BREANNA","JAYME","CARMEL","VERNELL","TOMASA","MANDI","DOMINGA","SANTA","MELODIE","LURA","ALEXA","TAMELA","RYAN","MIRNA","KERRIE","VENUS","NOEL","FELICITA","CRISTY","CARMELITA","BERNIECE","ANNEMARIE","TIARA","ROSEANNE","MISSY","CORI","ROXANA","PRICILLA","KRISTAL","JUNG","ELYSE","HAYDEE","ALETHA","BETTINA","MARGE","GILLIAN","FILOMENA","CHARLES","ZENAIDA","HARRIETTE","CARIDAD","VADA","UNA","ARETHA","PEARLINE","MARJORY","MARCELA","FLOR","EVETTE","ELOUISE","ALINA","TRINIDAD","DAVID","DAMARIS","CATHARINE","CARROLL","BELVA","NAKIA","MARLENA","LUANNE","LORINE","KARON","DORENE","DANITA","BRENNA","TATIANA","SAMMIE","LOUANN","LOREN","JULIANNA","ANDRIA","PHILOMENA","LUCILA","LEONORA","DOVIE","ROMONA","MIMI","JACQUELIN","GAYE","TONJA","MISTI","JOE","GENE","CHASTITY","STACIA","ROXANN","MICAELA","NIKITA","MEI","VELDA","MARLYS","JOHNNA","AURA","LAVERN","IVONNE","HAYLEY","NICKI","MAJORIE","HERLINDA","GEORGE","ALPHA","YADIRA","PERLA","GREGORIA","DANIEL","ANTONETTE","SHELLI","MOZELLE","MARIAH","JOELLE","CORDELIA","JOSETTE","CHIQUITA","TRISTA","LOUIS","LAQUITA","GEORGIANA","CANDI","SHANON","LONNIE","HILDEGARD","CECIL","VALENTINA","STEPHANY","MAGDA","KAROL","GERRY","GABRIELLA","TIANA","ROMA","RICHELLE","RAY","PRINCESS","OLETA","JACQUE","IDELLA","ALAINA","SUZANNA","JOVITA","BLAIR","TOSHA","RAVEN","NEREIDA","MARLYN","KYLA","JOSEPH","DELFINA","TENA","STEPHENIE","SABINA","NATHALIE","MARCELLE","GERTIE","DARLEEN","THEA","SHARONDA","SHANTEL","BELEN","VENESSA","ROSALINA","ONA","GENOVEVA","COREY","CLEMENTINE","ROSALBA","RENATE","RENATA","MI","IVORY","GEORGIANNA","FLOY","DORCAS","ARIANA","TYRA","THEDA","MARIAM","JULI","JESICA","DONNIE","VIKKI","VERLA","ROSELYN","MELVINA","JANNETTE","GINNY","DEBRAH","CORRIE","ASIA","VIOLETA","MYRTIS","LATRICIA","COLLETTE","CHARLEEN","ANISSA","VIVIANA","TWYLA","PRECIOUS","NEDRA","LATONIA","LAN","HELLEN","FABIOLA","ANNAMARIE","ADELL","SHARYN","CHANTAL","NIKI","MAUD","LIZETTE","LINDY","KIA","KESHA","JEANA","DANELLE","CHARLINE","CHANEL","CARROL","VALORIE","LIA","DORTHA","CRISTAL","SUNNY","LEONE","LEILANI","GERRI","DEBI","ANDRA","KESHIA","IMA","EULALIA","EASTER","DULCE","NATIVIDAD","LINNIE","KAMI","GEORGIE","CATINA","BROOK","ALDA","WINNIFRED","SHARLA","RUTHANN","MEAGHAN","MAGDALENE","LISSETTE","ADELAIDA","VENITA","TRENA","SHIRLENE","SHAMEKA","ELIZEBETH","DIAN","SHANTA","MICKEY","LATOSHA","CARLOTTA","WINDY","SOON","ROSINA","MARIANN","LEISA","JONNIE","DAWNA","CATHIE","BILLY","ASTRID","SIDNEY","LAUREEN","JANEEN","HOLLI","FAWN","VICKEY","TERESSA","SHANTE","RUBYE","MARCELINA","CHANDA","CARY","TERESE","SCARLETT","MARTY","MARNIE","LULU","LISETTE","JENIFFER","ELENOR","DORINDA","DONITA","CARMAN","BERNITA","ALTAGRACIA","ALETA","ADRIANNA","ZORAIDA","RONNIE","NICOLA","LYNDSEY","KENDALL","JANINA","CHRISSY","AMI","STARLA","PHYLIS","PHUONG","KYRA","CHARISSE","BLANCH","SANJUANITA","RONA","NANCI","MARILEE","MARANDA","CORY","BRIGETTE","SANJUANA","MARITA","KASSANDRA","JOYCELYN","IRA","FELIPA","CHELSIE","BONNY","MIREYA","LORENZA","KYONG","ILEANA","CANDELARIA","TONY","TOBY","SHERIE","OK","MARK","LUCIE","LEATRICE","LAKESHIA","GERDA","EDIE","BAMBI","MARYLIN","LAVON","HORTENSE","GARNET","EVIE","TRESSA","SHAYNA","LAVINA","KYUNG","JEANETTA","SHERRILL","SHARA","PHYLISS","MITTIE","ANABEL","ALESIA","THUY","TAWANDA","RICHARD","JOANIE","TIFFANIE","LASHANDA","KARISSA","ENRIQUETA","DARIA","DANIELLA","CORINNA","ALANNA","ABBEY","ROXANE","ROSEANNA","MAGNOLIA","LIDA","KYLE","JOELLEN","ERA","CORAL","CARLEEN","TRESA","PEGGIE","NOVELLA","NILA","MAYBELLE","JENELLE","CARINA","NOVA","MELINA","MARQUERITE","MARGARETTE","JOSEPHINA","EVONNE","DEVIN","CINTHIA","ALBINA","TOYA","TAWNYA","SHERITA","SANTOS","MYRIAM","LIZABETH","LISE","KEELY","JENNI","GISELLE","CHERYLE","ARDITH","ARDIS","ALESHA","ADRIANE","SHAINA","LINNEA","KAROLYN","HONG","FLORIDA","FELISHA","DORI","DARCI","ARTIE","ARMIDA","ZOLA","XIOMARA","VERGIE","SHAMIKA","NENA","NANNETTE","MAXIE","LOVIE","JEANE","JAIMIE","INGE","FARRAH","ELAINA","CAITLYN","STARR","FELICITAS","CHERLY","CARYL","YOLONDA","YASMIN","TEENA","PRUDENCE","PENNIE","NYDIA","MACKENZIE","ORPHA","MARVEL","LIZBETH","LAURETTE","JERRIE","HERMELINDA","CAROLEE","TIERRA","MIRIAN","META","MELONY","KORI","JENNETTE","JAMILA","ENA","ANH","YOSHIKO","SUSANNAH","SALINA","RHIANNON","JOLEEN","CRISTINE","ASHTON","ARACELY","TOMEKA","SHALONDA","MARTI","LACIE","KALA","JADA","ILSE","HAILEY","BRITTANI","ZONA","SYBLE","SHERRYL","RANDY","NIDIA","MARLO","KANDICE","KANDI","DEB","DEAN","AMERICA","ALYCIA","TOMMY","RONNA","NORENE","MERCY","JOSE","INGEBORG","GIOVANNA","GEMMA","CHRISTEL","AUDRY","ZORA","VITA","VAN","TRISH","STEPHAINE","SHIRLEE","SHANIKA","MELONIE","MAZIE","JAZMIN","INGA","HOA","HETTIE","GERALYN","FONDA","ESTRELLA","ADELLA","SU","SARITA","RINA","MILISSA","MARIBETH","GOLDA","EVON","ETHELYN","ENEDINA","CHERISE","CHANA","VELVA","TAWANNA","SADE","MIRTA","LI","KARIE","JACINTA","ELNA","DAVINA","CIERRA","ASHLIE","ALBERTHA","TANESHA","STEPHANI","NELLE","MINDI","LU","LORINDA","LARUE","FLORENE","DEMETRA","DEDRA","CIARA","CHANTELLE","ASHLY","SUZY","ROSALVA","NOELIA","LYDA","LEATHA","KRYSTYNA","KRISTAN","KARRI","DARLINE","DARCIE","CINDA","CHEYENNE","CHERRIE","AWILDA","ALMEDA","ROLANDA","LANETTE","JERILYN","GISELE","EVALYN","CYNDI","CLETA","CARIN","ZINA","ZENA","VELIA","TANIKA","PAUL","CHARISSA","THOMAS","TALIA","MARGARETE","LAVONDA","KAYLEE","KATHLENE","JONNA","IRENA","ILONA","IDALIA","CANDIS","CANDANCE","BRANDEE","ANITRA","ALIDA","SIGRID","NICOLETTE","MARYJO","LINETTE","HEDWIG","CHRISTIANA","CASSIDY","ALEXIA","TRESSIE","MODESTA","LUPITA","LITA","GLADIS","EVELIA","DAVIDA","CHERRI","CECILY","ASHELY","ANNABEL","AGUSTINA","WANITA","SHIRLY","ROSAURA","HULDA","EUN","BAILEY","YETTA","VERONA","THOMASINA","SIBYL","SHANNAN","MECHELLE","LUE","LEANDRA","LANI","KYLEE","KANDY","JOLYNN","FERNE","EBONI","CORENE","ALYSIA","ZULA","NADA","MOIRA","LYNDSAY","LORRETTA","JUAN","JAMMIE","HORTENSIA","GAYNELL","CAMERON","ADRIA","VINA","VICENTA","TANGELA","STEPHINE","NORINE","NELLA","LIANA","LESLEE","KIMBERELY","ILIANA","GLORY","FELICA","EMOGENE","ELFRIEDE","EDEN","EARTHA","CARMA","BEA","OCIE","MARRY","LENNIE","KIARA","JACALYN","CARLOTA","ARIELLE","YU","STAR","OTILIA","KIRSTIN","KACEY","JOHNETTA","JOEY","JOETTA","JERALDINE","JAUNITA","ELANA","DORTHEA","CAMI","AMADA","ADELIA","VERNITA","TAMAR","SIOBHAN","RENEA","RASHIDA","OUIDA","ODELL","NILSA","MERYL","KRISTYN","JULIETA","DANICA","BREANNE","AUREA","ANGLEA","SHERRON","ODETTE","MALIA","LORELEI","LIN","LEESA","KENNA","KATHLYN","FIONA","CHARLETTE","SUZIE","SHANTELL","SABRA","RACQUEL","MYONG","MIRA","MARTINE","LUCIENNE","LAVADA","JULIANN","JOHNIE","ELVERA","DELPHIA","CLAIR","CHRISTIANE","CHAROLETTE","CARRI","AUGUSTINE","ASHA","ANGELLA","PAOLA","NINFA","LEDA","LAI","EDA","SUNSHINE","STEFANI","SHANELL","PALMA","MACHELLE","LISSA","KECIA","KATHRYNE","KARLENE","JULISSA","JETTIE","JENNIFFER","HUI","CORRINA","CHRISTOPHER","CAROLANN","ALENA","TESS","ROSARIA","MYRTICE","MARYLEE","LIANE","KENYATTA","JUDIE","JANEY","IN","ELMIRA","ELDORA","DENNA","CRISTI","CATHI","ZAIDA","VONNIE","VIVA","VERNIE","ROSALINE","MARIELA","LUCIANA","LESLI","KARAN","FELICE","DENEEN","ADINA","WYNONA","TARSHA","SHERON","SHASTA","SHANITA","SHANI","SHANDRA","RANDA","PINKIE","PARIS","NELIDA","MARILOU","LYLA","LAURENE","LACI","JOI","JANENE","DOROTHA","DANIELE","DANI","CAROLYNN","CARLYN","BERENICE","AYESHA","ANNELIESE","ALETHEA","THERSA","TAMIKO","RUFINA","OLIVA","MOZELL","MARYLYN","MADISON","KRISTIAN","KATHYRN","KASANDRA","KANDACE","JANAE","GABRIEL","DOMENICA","DEBBRA","DANNIELLE","CHUN","BUFFY","BARBIE","ARCELIA","AJA","ZENOBIA","SHAREN","SHAREE","PATRICK","PAGE","MY","LAVINIA","KUM","KACIE","JACKELINE","HUONG","FELISA","EMELIA","ELEANORA","CYTHIA","CRISTIN","CLYDE","CLARIBEL","CARON","ANASTACIA","ZULMA","ZANDRA","YOKO","TENISHA","SUSANN","SHERILYN","SHAY","SHAWANDA","SABINE","ROMANA","MATHILDA","LINSEY","KEIKO","JOANA","ISELA","GRETTA","GEORGETTA","EUGENIE","DUSTY","DESIRAE","DELORA","CORAZON","ANTONINA","ANIKA","WILLENE","TRACEE","TAMATHA","REGAN","NICHELLE","MICKIE","MAEGAN","LUANA","LANITA","KELSIE","EDELMIRA","BREE","AFTON","TEODORA","TAMIE","SHENA","MEG","LINH","KELI","KACI","DANYELLE","BRITT","ARLETTE","ALBERTINE","ADELLE","TIFFINY","STORMY","SIMONA","NUMBERS","NICOLASA","NICHOL","NIA","NAKISHA","MEE","MAIRA","LOREEN","KIZZY","JOHNNY","JAY","FALLON","CHRISTENE","BOBBYE","ANTHONY","YING","VINCENZA","TANJA","RUBIE","RONI","QUEENIE","MARGARETT","KIMBERLI","IRMGARD","IDELL","HILMA","EVELINA","ESTA","EMILEE","DENNISE","DANIA","CARL","CARIE","ANTONIO","WAI","SANG","RISA","RIKKI","PARTICIA","MUI","MASAKO","MARIO","LUVENIA","LOREE","LONI","LIEN","KEVIN","GIGI","FLORENCIA","DORIAN","DENITA","DALLAS","CHI","BILLYE","ALEXANDER","TOMIKA","SHARITA","RANA","NIKOLE","NEOMA","MARGARITE","MADALYN","LUCINA","LAILA","KALI","JENETTE","GABRIELE","EVELYNE","ELENORA","CLEMENTINA","ALEJANDRINA","ZULEMA","VIOLETTE","VANNESSA","THRESA","RETTA","PIA","PATIENCE","NOELLA","NICKIE","JONELL","DELTA","CHUNG","CHAYA","CAMELIA","BETHEL","ANYA","ANDREW","THANH","SUZANN","SPRING","SHU","MILA","LILLA","LAVERNA","KEESHA","KATTIE","GIA","GEORGENE","EVELINE","ESTELL","ELIZBETH","VIVIENNE","VALLIE","TRUDIE","STEPHANE","MICHEL","MAGALY","MADIE","KENYETTA","KARREN","JANETTA","HERMINE","HARMONY","DRUCILLA","DEBBI","CELESTINA","CANDIE","BRITNI","BECKIE","AMINA","ZITA","YUN","YOLANDE","VIVIEN","VERNETTA","TRUDI","SOMMER","PEARLE","PATRINA","OSSIE","NICOLLE","LOYCE","LETTY","LARISA","KATHARINA","JOSELYN","JONELLE","JENELL","IESHA","HEIDE","FLORINDA","FLORENTINA","FLO","ELODIA","DORINE","BRUNILDA","BRIGID","ASHLI","ARDELLA","TWANA","THU","TARAH","SUNG","SHEA","SHAVON","SHANE","SERINA","RAYNA","RAMONITA","NGA","MARGURITE","LUCRECIA","KOURTNEY","KATI","JESUS","JESENIA","DIAMOND","CRISTA","AYANA","ALICA","ALIA","VINNIE","SUELLEN","ROMELIA","RACHELL","PIPER","OLYMPIA","MICHIKO","KATHALEEN","JOLIE","JESSI","JANESSA","HANA","HA","ELEASE","CARLETTA","BRITANY","SHONA","SALOME","ROSAMOND","REGENA","RAINA","NGOC","NELIA","LOUVENIA","LESIA","LATRINA","LATICIA","LARHONDA","JINA","JACKI","HOLLIS","HOLLEY","EMMY","DEEANN","CORETTA","ARNETTA","VELVET","THALIA","SHANICE","NETA","MIKKI","MICKI","LONNA","LEANA","LASHUNDA","KILEY","JOYE","JACQULYN","IGNACIA","HYUN","HIROKO","HENRY","HENRIETTE","ELAYNE","DELINDA","DARNELL","DAHLIA","COREEN","CONSUELA","CONCHITA","CELINE","BABETTE","AYANNA","ANETTE","ALBERTINA","SKYE","SHAWNEE","SHANEKA","QUIANA","PAMELIA","MIN","MERRI","MERLENE","MARGIT","KIESHA","KIERA","KAYLENE","JODEE","JENISE","ERLENE","EMMIE","ELSE","DARYL","DALILA","DAISEY","CODY","CASIE","BELIA","BABARA","VERSIE","VANESA","SHELBA","SHAWNDA","SAM","NORMAN","NIKIA","NAOMA","MARNA","MARGERET","MADALINE","LAWANA","KINDRA","JUTTA","JAZMINE","JANETT","HANNELORE","GLENDORA","GERTRUD","GARNETT","FREEDA","FREDERICA","FLORANCE","FLAVIA","DENNIS","CARLINE","BEVERLEE","ANJANETTE","VALDA","TRINITY","TAMALA","STEVIE","SHONNA","SHA","SARINA","ONEIDA","MICAH","MERILYN","MARLEEN","LURLINE","LENNA","KATHERIN","JIN","JENI","HAE","GRACIA","GLADY","FARAH","ERIC","ENOLA","EMA","DOMINQUE","DEVONA","DELANA","CECILA","CAPRICE","ALYSHA","ALI","ALETHIA","VENA","THERESIA","TAWNY","SONG","SHAKIRA","SAMARA","SACHIKO","RACHELE","PAMELLA","NICKY","MARNI","MARIEL","MAREN","MALISA","LIGIA","LERA","LATORIA","LARAE","KIMBER","KATHERN","KAREY","JENNEFER","JANETH","HALINA","FREDIA","DELISA","DEBROAH","CIERA","CHIN","ANGELIKA","ANDREE","ALTHA","YEN","VIVAN","TERRESA","TANNA","SUK","SUDIE","SOO","SIGNE","SALENA","RONNI","REBBECCA","MYRTIE","MCKENZIE","MALIKA","MAIDA","LOAN","LEONARDA","KAYLEIGH","FRANCE","ETHYL","ELLYN","DAYLE","CAMMIE","BRITTNI","BIRGIT","AVELINA","ASUNCION","ARIANNA","AKIKO","VENICE","TYESHA","TONIE","TIESHA","TAKISHA","STEFFANIE","SINDY","SANTANA","MEGHANN","MANDA","MACIE","LADY","KELLYE","KELLEE","JOSLYN","JASON","INGER","INDIRA","GLINDA","GLENNIS","FERNANDA","FAUSTINA","ENEIDA","ELICIA","DOT","DIGNA","DELL","ARLETTA","ANDRE","WILLIA","TAMMARA","TABETHA","SHERRELL","SARI","REFUGIO","REBBECA","PAULETTA","NIEVES","NATOSHA","NAKITA","MAMMIE","KENISHA","KAZUKO","KASSIE","GARY","EARLEAN","DAPHINE","CORLISS","CLOTILDE","CAROLYNE","BERNETTA","AUGUSTINA","AUDREA","ANNIS","ANNABELL","YAN","TENNILLE","TAMICA","SELENE","SEAN","ROSANA","REGENIA","QIANA","MARKITA","MACY","LEEANNE","LAURINE","KYM","JESSENIA","JANITA","GEORGINE","GENIE","EMIKO","ELVIE","DEANDRA","DAGMAR","CORIE","COLLEN","CHERISH","ROMAINE","PORSHA","PEARLENE","MICHELINE","MERNA","MARGORIE","MARGARETTA","LORE","KENNETH","JENINE","HERMINA","FREDERICKA","ELKE","DRUSILLA","DORATHY","DIONE","DESIRE","CELENA","BRIGIDA","ANGELES","ALLEGRA","THEO","TAMEKIA","SYNTHIA","STEPHEN","SOOK","SLYVIA","ROSANN","REATHA","RAYE","MARQUETTA","MARGART","LING","LAYLA","KYMBERLY","KIANA","KAYLEEN","KATLYN","KARMEN","JOELLA","IRINA","EMELDA","ELENI","DETRA","CLEMMIE","CHERYLL","CHANTELL","CATHEY","ARNITA","ARLA","ANGLE","ANGELIC","ALYSE","ZOFIA","THOMASINE","TENNIE","SON","SHERLY","SHERLEY","SHARYL","REMEDIOS","PETRINA","NICKOLE","MYUNG","MYRLE","MOZELLA","LOUANNE","LISHA","LATIA","LANE","KRYSTA","JULIENNE","JOEL","JEANENE","JACQUALINE","ISAURA","GWENDA","EARLEEN","DONALD","CLEOPATRA","CARLIE","AUDIE","ANTONIETTA","ALISE","ALEX","VERDELL","VAL","TYLER","TOMOKO","THAO","TALISHA","STEVEN","SO","SHEMIKA","SHAUN","SCARLET","SAVANNA","SANTINA","ROSIA","RAEANN","ODILIA","NANA","MINNA","MAGAN","LYNELLE","LE","KARMA","JOEANN","IVANA","INELL","ILANA","HYE","HONEY","HEE","GUDRUN","FRANK","DREAMA","CRISSY","CHANTE","CARMELINA","ARVILLA","ARTHUR","ANNAMAE","ALVERA","ALEIDA","AARON","YEE","YANIRA","VANDA","TIANNA","TAM","STEFANIA","SHIRA","PERRY","NICOL","NANCIE","MONSERRATE","MINH","MELYNDA","MELANY","MATTHEW","LOVELLA","LAURE","KIRBY","KACY","JACQUELYNN","HYON","GERTHA","FRANCISCO","ELIANA","CHRISTENA","CHRISTEEN","CHARISE","CATERINA","CARLEY","CANDYCE","ARLENA","AMMIE","YANG","WILLETTE","VANITA","TUYET","TINY","SYREETA","SILVA","SCOTT","RONALD","PENNEY","NYLA","MICHAL","MAURICE","MARYAM","MARYA","MAGEN","LUDIE","LOMA","LIVIA","LANELL","KIMBERLIE","JULEE","DONETTA","DIEDRA","DENISHA","DEANE","DAWNE","CLARINE","CHERRYL","BRONWYN","BRANDON","ALLA","VALERY","TONDA","SUEANN","SORAYA","SHOSHANA","SHELA","SHARLEEN","SHANELLE","NERISSA","MICHEAL","MERIDITH","MELLIE","MAYE","MAPLE","MAGARET","LUIS","LILI","LEONILA","LEONIE","LEEANNA","LAVONIA","LAVERA","KRISTEL","KATHEY","KATHE","JUSTIN","JULIAN","JIMMY","JANN","ILDA","HILDRED","HILDEGARDE","GENIA","FUMIKO","EVELIN","ERMELINDA","ELLY","DUNG","DOLORIS","DIONNA","DANAE","BERNEICE","ANNICE","ALIX","VERENA","VERDIE","TRISTAN","SHAWNNA","SHAWANA","SHAUNNA","ROZELLA","RANDEE","RANAE","MILAGRO","LYNELL","LUISE","LOUIE","LOIDA","LISBETH","KARLEEN","JUNITA","JONA","ISIS","HYACINTH","HEDY","GWENN","ETHELENE","ERLINE","EDWARD","DONYA","DOMONIQUE","DELICIA","DANNETTE","CICELY","BRANDA","BLYTHE","BETHANN","ASHLYN","ANNALEE","ALLINE","YUKO","VELLA","TRANG","TOWANDA","TESHA","SHERLYN","NARCISA","MIGUELINA","MERI","MAYBELL","MARLANA","MARGUERITA","MADLYN","LUNA","LORY","LORIANN","LIBERTY","LEONORE","LEIGHANN","LAURICE","LATESHA","LARONDA","KATRICE","KASIE","KARL","KALEY","JADWIGA","GLENNIE","GEARLDINE","FRANCINA","EPIFANIA","DYAN","DORIE","DIEDRE","DENESE","DEMETRICE","DELENA","DARBY","CRISTIE","CLEORA","CATARINA","CARISA","BERNIE","BARBERA","ALMETA","TRULA","TEREASA","SOLANGE","SHEILAH","SHAVONNE","SANORA","ROCHELL","MATHILDE","MARGARETA","MAIA","LYNSEY","LAWANNA","LAUNA","KENA","KEENA","KATIA","JAMEY","GLYNDA","GAYLENE","ELVINA","ELANOR","DANUTA","DANIKA","CRISTEN","CORDIE","COLETTA","CLARITA","CARMON","BRYNN","AZUCENA","AUNDREA","ANGELE","YI","WALTER","VERLIE","VERLENE","TAMESHA","SILVANA","SEBRINA","SAMIRA","REDA","RAYLENE","PENNI","PANDORA","NORAH","NOMA","MIREILLE","MELISSIA","MARYALICE","LARAINE","KIMBERY","KARYL","KARINE","KAM","JOLANDA","JOHANA","JESUSA","JALEESA","JAE","JACQUELYNE","IRISH","ILUMINADA","HILARIA","HANH","GENNIE","FRANCIE","FLORETTA","EXIE","EDDA","DREMA","DELPHA","BEV","BARBAR","ASSUNTA","ARDELL","ANNALISA","ALISIA","YUKIKO","YOLANDO","WONDA","WEI","WALTRAUD","VETA","TEQUILA","TEMEKA","TAMEIKA","SHIRLEEN","SHENITA","PIEDAD","OZELLA","MIRTHA","MARILU","KIMIKO","JULIANE","JENICE","JEN","JANAY","JACQUILINE","HILDE","FE","FAE","EVAN","EUGENE","ELOIS","ECHO","DEVORAH","CHAU","BRINDA","BETSEY","ARMINDA","ARACELIS","APRYL","ANNETT","ALISHIA","VEOLA","USHA","TOSHIKO","THEOLA","TASHIA","TALITHA","SHERY","RUDY","RENETTA","REIKO","RASHEEDA","OMEGA","OBDULIA","MIKA","MELAINE","MEGGAN","MARTIN","MARLEN","MARGET","MARCELINE","MANA","MAGDALEN","LIBRADA","LEZLIE","LEXIE","LATASHIA","LASANDRA","KELLE","ISIDRA","ISA","INOCENCIA","GWYN","FRANCOISE","ERMINIA","ERINN","DIMPLE","DEVORA","CRISELDA","ARMANDA","ARIE","ARIANE","ANGELO","ANGELENA","ALLEN","ALIZA","ADRIENE","ADALINE","XOCHITL","TWANNA","TRAN","TOMIKO","TAMISHA","TAISHA","SUSY","SIU","RUTHA","ROXY","RHONA","RAYMOND","OTHA","NORIKO","NATASHIA","MERRIE","MELVIN","MARINDA","MARIKO","MARGERT","LORIS","LIZZETTE","LEISHA","KAILA","KA","JOANNIE","JERRICA","JENE","JANNET","JANEE","JACINDA","HERTA","ELENORE","DORETTA","DELAINE","DANIELL","CLAUDIE","CHINA","BRITTA","APOLONIA","AMBERLY","ALEASE","YURI","YUK","WEN","WANETA","UTE","TOMI","SHARRI","SANDIE","ROSELLE","REYNALDA","RAGUEL","PHYLICIA","PATRIA","OLIMPIA","ODELIA","MITZIE","MITCHELL","MISS","MINDA","MIGNON","MICA","MENDY","MARIVEL","MAILE","LYNETTA","LAVETTE","LAURYN","LATRISHA","LAKIESHA","KIERSTEN","KARY","JOSPHINE","JOLYN","JETTA","JANISE","JACQUIE","IVELISSE","GLYNIS","GIANNA","GAYNELLE","EMERALD","DEMETRIUS","DANYELL","DANILLE","DACIA","CORALEE","CHER","CEOLA","BRETT","BELL","ARIANNE","ALESHIA","YUNG","WILLIEMAE","TROY","TRINH","THORA","TAI","SVETLANA","SHERIKA","SHEMEKA","SHAUNDA","ROSELINE","RICKI","MELDA","MALLIE","LAVONNA","LATINA","LARRY","LAQUANDA","LALA","LACHELLE","KLARA","KANDIS","JOHNA","JEANMARIE","JAYE","HANG","GRAYCE","GERTUDE","EMERITA","EBONIE","CLORINDA","CHING","CHERY","CAROLA","BREANN","BLOSSOM","BERNARDINE","BECKI","ARLETHA","ARGELIA","ARA","ALITA","YULANDA","YON","YESSENIA","TOBI","TASIA","SYLVIE","SHIRL","SHIRELY","SHERIDAN","SHELLA","SHANTELLE","SACHA","ROYCE","REBECKA","REAGAN","PROVIDENCIA","PAULENE","MISHA","MIKI","MARLINE","MARICA","LORITA","LATOYIA","LASONYA","KERSTIN","KENDA","KEITHA","KATHRIN","JAYMIE","JACK","GRICELDA","GINETTE","ERYN","ELINA","ELFRIEDA","DANYEL","CHEREE","CHANELLE","BARRIE","AVERY","AURORE","ANNAMARIA","ALLEEN","AILENE","AIDE","YASMINE","VASHTI","VALENTINE","TREASA","TORY","TIFFANEY","SHERYLL","SHARIE","SHANAE","SAU","RAISA","PA","NEDA","MITSUKO","MIRELLA","MILDA","MARYANNA","MARAGRET","MABELLE","LUETTA","LORINA","LETISHA","LATARSHA","LANELLE","LAJUANA","KRISSY","KARLY","KARENA","JON","JESSIKA","JERICA","JEANELLE","JANUARY","JALISA","JACELYN","IZOLA","IVEY","GREGORY","EUNA","ETHA","DREW","DOMITILA","DOMINICA","DAINA","CREOLA","CARLI","CAMIE","BUNNY","BRITTNY","ASHANTI","ANISHA","ALEEN","ADAH","YASUKO","WINTER","VIKI","VALRIE","TONA","TINISHA","THI","TERISA","TATUM","TANEKA","SIMONNE","SHALANDA","SERITA","RESSIE","REFUGIA","PAZ","OLENE","NA","MERRILL","MARGHERITA","MANDIE","MAN","MAIRE","LYNDIA","LUCI","LORRIANE","LORETA","LEONIA","LAVONA","LASHAWNDA","LAKIA","KYOKO","KRYSTINA","KRYSTEN","KENIA","KELSI","JUDE","JEANICE","ISOBEL","GEORGIANN","GENNY","FELICIDAD","EILENE","DEON","DELOISE","DEEDEE","DANNIE","CONCEPTION","CLORA","CHERILYN","CHANG","CALANDRA","BERRY","ARMANDINA","ANISA","ULA","TIMOTHY","TIERA","THERESSA","STEPHANIA","SIMA","SHYLA","SHONTA","SHERA","SHAQUITA","SHALA","SAMMY","ROSSANA","NOHEMI","NERY","MORIAH","MELITA","MELIDA","MELANI","MARYLYNN","MARISHA","MARIETTE","MALORIE","MADELENE","LUDIVINA","LORIA","LORETTE","LORALEE","LIANNE","LEON","LAVENIA","LAURINDA","LASHON","KIT","KIMI","KEILA","KATELYNN","KAI","JONE","JOANE","JI","JAYNA","JANELLA","JA","HUE","HERTHA","FRANCENE","ELINORE","DESPINA","DELSIE","DEEDRA","CLEMENCIA","CARRY","CAROLIN","CARLOS","BULAH","BRITTANIE","BOK","BLONDELL","BIBI","BEAULAH","BEATA","ANNITA","AGRIPINA","VIRGEN","VALENE","UN","TWANDA","TOMMYE","TOI","TARRA","TARI","TAMMERA","SHAKIA","SADYE","RUTHANNE","ROCHEL","RIVKA","PURA","NENITA","NATISHA","MING","MERRILEE","MELODEE","MARVIS","LUCILLA","LEENA","LAVETA","LARITA","LANIE","KEREN","ILEEN","GEORGEANN","GENNA","GENESIS","FRIDA","EWA","EUFEMIA","EMELY","ELA","EDYTH","DEONNA","DEADRA","DARLENA","CHANELL","CHAN","CATHERN","CASSONDRA","CASSAUNDRA","BERNARDA","BERNA","ARLINDA","ANAMARIA","ALBERT","WESLEY","VERTIE","VALERI","TORRI","TATYANA","STASIA","SHERISE","SHERILL","SEASON","SCOTTIE","SANDA","RUTHE","ROSY","ROBERTO","ROBBI","RANEE","QUYEN","PEARLY","PALMIRA","ONITA","NISHA","NIESHA","NIDA","NEVADA","NAM","MERLYN","MAYOLA","MARYLOUISE","MARYLAND","MARX","MARTH","MARGENE","MADELAINE","LONDA","LEONTINE","LEOMA","LEIA","LAWRENCE","LAURALEE","LANORA","LAKITA","KIYOKO","KETURAH","KATELIN","KAREEN","JONIE","JOHNETTE","JENEE","JEANETT","IZETTA","HIEDI","HEIKE","HASSIE","HAROLD","GIUSEPPINA","GEORGANN","FIDELA","FERNANDE","ELWANDA","ELLAMAE","ELIZ","DUSTI","DOTTY","CYNDY","CORALIE","CELESTA","ARGENTINA","ALVERTA","XENIA","WAVA","VANETTA","TORRIE","TASHINA","TANDY","TAMBRA","TAMA","STEPANIE","SHILA","SHAUNTA","SHARAN","SHANIQUA","SHAE","SETSUKO","SERAFINA","SANDEE","ROSAMARIA","PRISCILA","OLINDA","NADENE","MUOI","MICHELINA","MERCEDEZ","MARYROSE","MARIN","MARCENE","MAO","MAGALI","MAFALDA","LOGAN","LINN","LANNIE","KAYCE","KAROLINE","KAMILAH","KAMALA","JUSTA","JOLINE","JENNINE","JACQUETTA","IRAIDA","GERALD","GEORGEANNA","FRANCHESCA","FAIRY","EMELINE","ELANE","EHTEL","EARLIE","DULCIE","DALENE","CRIS","CLASSIE","CHERE","CHARIS","CAROYLN","CARMINA","CARITA","BRIAN","BETHANIE","AYAKO","ARICA","AN","ALYSA","ALESSANDRA","AKILAH","ADRIEN","ZETTA","YOULANDA","YELENA","YAHAIRA","XUAN","WENDOLYN","VICTOR","TIJUANA","TERRELL","TERINA","TERESIA","SUZI","SUNDAY","SHERELL","SHAVONDA","SHAUNTE","SHARDA","SHAKITA","SENA","RYANN","RUBI","RIVA","REGINIA","REA","RACHAL","PARTHENIA","PAMULA","MONNIE","MONET","MICHAELE","MELIA","MARINE","MALKA","MAISHA","LISANDRA","LEO","LEKISHA","LEAN","LAURENCE","LAKENDRA","KRYSTIN","KORTNEY","KIZZIE","KITTIE","KERA","KENDAL","KEMBERLY","KANISHA","JULENE","JULE","JOSHUA","JOHANNE","JEFFREY","JAMEE","HAN","HALLEY","GIDGET","GALINA","FREDRICKA","FLETA","FATIMAH","EUSEBIA","ELZA","ELEONORE","DORTHEY","DORIA","DONELLA","DINORAH","DELORSE","CLARETHA","CHRISTINIA","CHARLYN","BONG","BELKIS","AZZIE","ANDERA","AIKO","ADENA","YER","YAJAIRA","WAN","VANIA","ULRIKE","TOSHIA","TIFANY","STEFANY","SHIZUE","SHENIKA","SHAWANNA","SHAROLYN","SHARILYN","SHAQUANA","SHANTAY","SEE","ROZANNE","ROSELEE","RICKIE","REMONA","REANNA","RAELENE","QUINN","PHUNG","PETRONILA","NATACHA","NANCEY","MYRL","MIYOKO","MIESHA","MERIDETH","MARVELLA","MARQUITTA","MARHTA","MARCHELLE","LIZETH","LIBBIE","LAHOMA","LADAWN","KINA","KATHELEEN","KATHARYN","KARISA","KALEIGH","JUNIE","JULIEANN","JOHNSIE","JANEAN","JAIMEE","JACKQUELINE","HISAKO","HERMA","HELAINE","GWYNETH","GLENN","GITA","EUSTOLIA","EMELINA","ELIN","EDRIS","DONNETTE","DONNETTA","DIERDRE","DENAE","DARCEL","CLAUDE","CLARISA","CINDERELLA","CHIA","CHARLESETTA","CHARITA","CELSA","CASSY","CASSI","CARLEE","BRUNA","BRITTANEY","BRANDE","BILLI","BAO","ANTONETTA","ANGLA","ANGELYN","ANALISA","ALANE","WENONA","WENDIE","VERONIQUE","VANNESA","TOBIE","TEMPIE","SUMIKO","SULEMA","SPARKLE","SOMER","SHEBA","SHAYNE","SHARICE","SHANEL","SHALON","SAGE","ROY","ROSIO","ROSELIA","RENAY","REMA","REENA","PORSCHE","PING","PEG","OZIE","ORETHA","ORALEE","ODA","NU","NGAN","NAKESHA","MILLY","MARYBELLE","MARLIN","MARIS","MARGRETT","MARAGARET","MANIE","LURLENE","LILLIA","LIESELOTTE","LAVELLE","LASHAUNDA","LAKEESHA","KEITH","KAYCEE","KALYN","JOYA","JOETTE","JENAE","JANIECE","ILLA","GRISEL","GLAYDS","GENEVIE","GALA","FREDDA","FRED","ELMER","ELEONOR","DEBERA","DEANDREA","DAN","CORRINNE","CORDIA","CONTESSA","COLENE","CLEOTILDE","CHARLOTT","CHANTAY","CECILLE","BEATRIS","AZALEE","ARLEAN","ARDATH","ANJELICA","ANJA","ALFREDIA","ALEISHA","ADAM","ZADA","YUONNE","XIAO","WILLODEAN","WHITLEY","VENNIE","VANNA","TYISHA","TOVA","TORIE","TONISHA","TILDA","TIEN","TEMPLE","SIRENA","SHERRIL","SHANTI","SHAN","SENAIDA","SAMELLA","ROBBYN","RENDA","REITA","PHEBE","PAULITA","NOBUKO","NGUYET","NEOMI","MOON","MIKAELA","MELANIA","MAXIMINA","MARG","MAISIE","LYNNA","LILLI","LAYNE","LASHAUN","LAKENYA","LAEL","KIRSTIE","KATHLINE","KASHA","KARLYN","KARIMA","JOVAN","JOSEFINE","JENNELL","JACQUI","JACKELYN","HYO","HIEN","GRAZYNA","FLORRIE","FLORIA","ELEONORA","DWANA","DORLA","DONG","DELMY","DEJA","DEDE","DANN","CRYSTA","CLELIA","CLARIS","CLARENCE","CHIEKO","CHERLYN","CHERELLE","CHARMAIN","CHARA","CAMMY","BEE","ARNETTE","ARDELLE","ANNIKA","AMIEE","AMEE","ALLENA","YVONE","YUKI","YOSHIE","YEVETTE","YAEL","WILLETTA","VONCILE","VENETTA","TULA","TONETTE","TIMIKA","TEMIKA","TELMA","TEISHA","TAREN","TA","STACEE","SHIN","SHAWNTA","SATURNINA","RICARDA","POK","PASTY","ONIE","NUBIA","MORA","MIKE","MARIELLE","MARIELLA","MARIANELA","MARDELL","MANY","LUANNA","LOISE","LISABETH","LINDSY","LILLIANA","LILLIAM","LELAH","LEIGHA","LEANORA","LANG","KRISTEEN","KHALILAH","KEELEY","KANDRA","JUNKO","JOAQUINA","JERLENE","JANI","JAMIKA","JAME","HSIU","HERMILA","GOLDEN","GENEVIVE","EVIA","EUGENA","EMMALINE","ELFREDA","ELENE","DONETTE","DELCIE","DEEANNA","DARCEY","CUC","CLARINDA","CIRA","CHAE","CELINDA","CATHERYN","CATHERIN","CASIMIRA","CARMELIA","CAMELLIA","BREANA","BOBETTE","BERNARDINA","BEBE","BASILIA","ARLYNE","AMAL","ALAYNA","ZONIA","ZENIA","YURIKO","YAEKO","WYNELL","WILLOW","WILLENA","VERNIA","TU","TRAVIS","TORA","TERRILYN","TERICA","TENESHA","TAWNA","TAJUANA","TAINA","STEPHNIE","SONA","SOL","SINA","SHONDRA","SHIZUKO","SHERLENE","SHERICE","SHARIKA","ROSSIE","ROSENA","RORY","RIMA","RIA","RHEBA","RENNA","PETER","NATALYA","NANCEE","MELODI","MEDA","MAXIMA","MATHA","MARKETTA","MARICRUZ","MARCELENE","MALVINA","LUBA","LOUETTA","LEIDA","LECIA","LAURAN","LASHAWNA","LAINE","KHADIJAH","KATERINE","KASI","KALLIE","JULIETTA","JESUSITA","JESTINE","JESSIA","JEREMY","JEFFIE","JANYCE","ISADORA","GEORGIANNE","FIDELIA","EVITA","EURA","EULAH","ESTEFANA","ELSY","ELIZABET","ELADIA","DODIE","DION","DIA","DENISSE","DELORAS","DELILA","DAYSI","DAKOTA","CURTIS","CRYSTLE","CONCHA","COLBY","CLARETTA","CHU","CHRISTIA","CHARLSIE","CHARLENA","CARYLON","BETTYANN","ASLEY","ASHLEA","AMIRA","AI","AGUEDA","AGNUS","YUETTE","VINITA","VICTORINA","TYNISHA","TREENA","TOCCARA","TISH","THOMASENA","TEGAN","SOILA","SHILOH","SHENNA","SHARMAINE","SHANTAE","SHANDI","SEPTEMBER","SARAN","SARAI","SANA","SAMUEL","SALLEY","ROSETTE","ROLANDE","REGINE","OTELIA","OSCAR","OLEVIA","NICHOLLE","NECOLE","NAIDA","MYRTA","MYESHA","MITSUE","MINTA","MERTIE","MARGY","MAHALIA","MADALENE","LOVE","LOURA","LOREAN","LEWIS","LESHA","LEONIDA","LENITA","LAVONE","LASHELL","LASHANDRA","LAMONICA","KIMBRA","KATHERINA","KARRY","KANESHA","JULIO","JONG","JENEVA","JAQUELYN","HWA","GILMA","GHISLAINE","GERTRUDIS","FRANSISCA","FERMINA","ETTIE","ETSUKO","ELLIS","ELLAN","ELIDIA","EDRA","DORETHEA","DOREATHA","DENYSE","DENNY","DEETTA","DAINE","CYRSTAL","CORRIN","CAYLA","CARLITA","CAMILA","BURMA","BULA","BUENA","BLAKE","BARABARA","AVRIL","AUSTIN","ALAINE","ZANA","WILHEMINA","WANETTA","VIRGIL","VI","VERONIKA","VERNON","VERLINE","VASILIKI","TONITA","TISA","TEOFILA","TAYNA","TAUNYA","TANDRA","TAKAKO","SUNNI","SUANNE","SIXTA","SHARELL","SEEMA","RUSSELL","ROSENDA","ROBENA","RAYMONDE","PEI","PAMILA","OZELL","NEIDA","NEELY","MISTIE","MICHA","MERISSA","MAURITA","MARYLN","MARYETTA","MARSHALL","MARCELL","MALENA","MAKEDA","MADDIE","LOVETTA","LOURIE","LORRINE","LORILEE","LESTER","LAURENA","LASHAY","LARRAINE","LAREE","LACRESHA","KRISTLE","KRISHNA","KEVA","KEIRA","KAROLE","JOIE","JINNY","JEANNETTA","JAMA","HEIDY","GILBERTE","GEMA","FAVIOLA","EVELYNN","ENDA","ELLI","ELLENA","DIVINA","DAGNY","COLLENE","CODI","CINDIE","CHASSIDY","CHASIDY","CATRICE","CATHERINA","CASSEY","CAROLL","CARLENA","CANDRA","CALISTA","BRYANNA","BRITTENY","BEULA","BARI","AUDRIE","AUDRIA","ARDELIA","ANNELLE","ANGILA","ALONA","ALLYN","DOUGLAS","ROGER","JONATHAN","RALPH","NICHOLAS","BENJAMIN","BRUCE","HARRY","WAYNE","STEVE","HOWARD","ERNEST","PHILLIP","TODD","CRAIG","ALAN","PHILIP","EARL","DANNY","BRYAN","STANLEY","LEONARD","NATHAN","MANUEL","RODNEY","MARVIN","VINCENT","JEFFERY","JEFF","CHAD","JACOB","ALFRED","BRADLEY","HERBERT","FREDERICK","EDWIN","DON","RICKY","RANDALL","BARRY","BERNARD","LEROY","MARCUS","THEODORE","CLIFFORD","MIGUEL","JIM","TOM","CALVIN","BILL","LLOYD","DEREK","WARREN","DARRELL","JEROME","FLOYD","ALVIN","TIM","GORDON","GREG","JORGE","DUSTIN","PEDRO","DERRICK","ZACHARY","HERMAN","GLEN","HECTOR","RICARDO","RICK","BRENT","RAMON","GILBERT","MARC","REGINALD","RUBEN","NATHANIEL","RAFAEL","EDGAR","MILTON","RAUL","BEN","CHESTER","DUANE","FRANKLIN","BRAD","RON","ROLAND","ARNOLD","HARVEY","JARED","ERIK","DARRYL","NEIL","JAVIER","FERNANDO","CLINTON","TED","MATHEW","TYRONE","DARREN","LANCE","KURT","ALLAN","NELSON","GUY","CLAYTON","HUGH","MAX","DWAYNE","DWIGHT","ARMANDO","FELIX","EVERETT","IAN","WALLACE","KEN","BOB","ALFREDO","ALBERTO","DAVE","IVAN","BYRON","ISAAC","MORRIS","CLIFTON","WILLARD","ROSS","ANDY","SALVADOR","KIRK","SERGIO","SETH","KENT","TERRANCE","EDUARDO","TERRENCE","ENRIQUE","WADE","STUART","FREDRICK","ARTURO","ALEJANDRO","NICK","LUTHER","WENDELL","JEREMIAH","JULIUS","OTIS","TREVOR","OLIVER","LUKE","HOMER","GERARD","DOUG","KENNY","HUBERT","LYLE","MATT","ALFONSO","ORLANDO","REX","CARLTON","ERNESTO","NEAL","PABLO","LORENZO","OMAR","WILBUR","GRANT","HORACE","RODERICK","ABRAHAM","WILLIS","RICKEY","ANDRES","CESAR","JOHNATHAN","MALCOLM","RUDOLPH","DAMON","KELVIN","PRESTON","ALTON","ARCHIE","MARCO","WM","PETE","RANDOLPH","GARRY","GEOFFREY","JONATHON","FELIPE","GERARDO","ED","DOMINIC","DELBERT","COLIN","GUILLERMO","EARNEST","LUCAS","BENNY","SPENCER","RODOLFO","MYRON","EDMUND","GARRETT","SALVATORE","CEDRIC","LOWELL","GREGG","SHERMAN","WILSON","SYLVESTER","ROOSEVELT","ISRAEL","JERMAINE","FORREST","WILBERT","LELAND","SIMON","CLARK","IRVING","BRYANT","OWEN","RUFUS","WOODROW","KRISTOPHER","MACK","LEVI","MARCOS","GUSTAVO","JAKE","LIONEL","GILBERTO","CLINT","NICOLAS","ISMAEL","ORVILLE","ERVIN","DEWEY","AL","WILFRED","JOSH","HUGO","IGNACIO","CALEB","TOMAS","SHELDON","ERICK","STEWART","DOYLE","DARREL","ROGELIO","TERENCE","SANTIAGO","ALONZO","ELIAS","BERT","ELBERT","RAMIRO","CONRAD","NOAH","GRADY","PHIL","CORNELIUS","LAMAR","ROLANDO","CLAY","PERCY","DEXTER","BRADFORD","DARIN","AMOS","MOSES","IRVIN","SAUL","ROMAN","RANDAL","TIMMY","DARRIN","WINSTON","BRENDAN","ABEL","DOMINICK","BOYD","EMILIO","ELIJAH","DOMINGO","EMMETT","MARLON","EMANUEL","JERALD","EDMOND","EMIL","DEWAYNE","WILL","OTTO","TEDDY","REYNALDO","BRET","JESS","TRENT","HUMBERTO","EMMANUEL","STEPHAN","VICENTE","LAMONT","GARLAND","MILES","EFRAIN","HEATH","RODGER","HARLEY","ETHAN","ELDON","ROCKY","PIERRE","JUNIOR","FREDDY","ELI","BRYCE","ANTOINE","STERLING","CHASE","GROVER","ELTON","CLEVELAND","DYLAN","CHUCK","DAMIAN","REUBEN","STAN","AUGUST","LEONARDO","JASPER","RUSSEL","ERWIN","BENITO","HANS","MONTE","BLAINE","ERNIE","CURT","QUENTIN","AGUSTIN","MURRAY","JAMAL","ADOLFO","HARRISON","TYSON","BURTON","BRADY","ELLIOTT","WILFREDO","BART","JARROD","VANCE","DENIS","DAMIEN","JOAQUIN","HARLAN","DESMOND","ELLIOT","DARWIN","GREGORIO","BUDDY","XAVIER","KERMIT","ROSCOE","ESTEBAN","ANTON","SOLOMON","SCOTTY","NORBERT","ELVIN","WILLIAMS","NOLAN","ROD","QUINTON","HAL","BRAIN","ROB","ELWOOD","KENDRICK","DARIUS","MOISES","FIDEL","THADDEUS","CLIFF","MARCEL","JACKSON","RAPHAEL","BRYON","ARMAND","ALVARO","JEFFRY","DANE","JOESPH","THURMAN","NED","RUSTY","MONTY","FABIAN","REGGIE","MASON","GRAHAM","ISAIAH","VAUGHN","GUS","LOYD","DIEGO","ADOLPH","NORRIS","MILLARD","ROCCO","GONZALO","DERICK","RODRIGO","WILEY","RIGOBERTO","ALPHONSO","TY","NOE","VERN","REED","JEFFERSON","ELVIS","BERNARDO","MAURICIO","HIRAM","DONOVAN","BASIL","RILEY","NICKOLAS","MAYNARD","SCOT","VINCE","QUINCY","EDDY","SEBASTIAN","FEDERICO","ULYSSES","HERIBERTO","DONNELL","COLE","DAVIS","GAVIN","EMERY","WARD","ROMEO","JAYSON","DANTE","CLEMENT","COY","MAXWELL","JARVIS","BRUNO","ISSAC","DUDLEY","BROCK","SANFORD","CARMELO","BARNEY","NESTOR","STEFAN","DONNY","ART","LINWOOD","BEAU","WELDON","GALEN","ISIDRO","TRUMAN","DELMAR","JOHNATHON","SILAS","FREDERIC","DICK","IRWIN","MERLIN","CHARLEY","MARCELINO","HARRIS","CARLO","TRENTON","KURTIS","HUNTER","AURELIO","WINFRED","VITO","COLLIN","DENVER","CARTER","LEONEL","EMORY","PASQUALE","MOHAMMAD","MARIANO","DANIAL","LANDON","DIRK","BRANDEN","ADAN","BUFORD","GERMAN","WILMER","EMERSON","ZACHERY","FLETCHER","JACQUES","ERROL","DALTON","MONROE","JOSUE","EDWARDO","BOOKER","WILFORD","SONNY","SHELTON","CARSON","THERON","RAYMUNDO","DAREN","HOUSTON","ROBBY","LINCOLN","GENARO","BENNETT","OCTAVIO","CORNELL","HUNG","ARRON","ANTONY","HERSCHEL","GIOVANNI","GARTH","CYRUS","CYRIL","RONNY","LON","FREEMAN","DUNCAN","KENNITH","CARMINE","ERICH","CHADWICK","WILBURN","RUSS","REID","MYLES","ANDERSON","MORTON","JONAS","FOREST","MITCHEL","MERVIN","ZANE","RICH","JAMEL","LAZARO","ALPHONSE","RANDELL","MAJOR","JARRETT","BROOKS","ABDUL","LUCIANO","SEYMOUR","EUGENIO","MOHAMMED","VALENTIN","CHANCE","ARNULFO","LUCIEN","FERDINAND","THAD","EZRA","ALDO","RUBIN","ROYAL","MITCH","EARLE","ABE","WYATT","MARQUIS","LANNY","KAREEM","JAMAR","BORIS","ISIAH","EMILE","ELMO","ARON","LEOPOLDO","EVERETTE","JOSEF","ELOY","RODRICK","REINALDO","LUCIO","JERROD","WESTON","HERSHEL","BARTON","PARKER","LEMUEL","BURT","JULES","GIL","ELISEO","AHMAD","NIGEL","EFREN","ANTWAN","ALDEN","MARGARITO","COLEMAN","DINO","OSVALDO","LES","DEANDRE","NORMAND","KIETH","TREY","NORBERTO","NAPOLEON","JEROLD","FRITZ","ROSENDO","MILFORD","CHRISTOPER","ALFONZO","LYMAN","JOSIAH","BRANT","WILTON","RICO","JAMAAL","DEWITT","BRENTON","OLIN","FOSTER","FAUSTINO","CLAUDIO","JUDSON","GINO","EDGARDO","ALEC","TANNER","JARRED","DONN","TAD","PRINCE","PORFIRIO","ODIS","LENARD","CHAUNCEY","TOD","MEL","MARCELO","KORY","AUGUSTUS","KEVEN","HILARIO","BUD","SAL","ORVAL","MAURO","ZACHARIAH","OLEN","ANIBAL","MILO","JED","DILLON","AMADO","NEWTON","LENNY","RICHIE","HORACIO","BRICE","MOHAMED","DELMER","DARIO","REYES","MAC","JONAH","JERROLD","ROBT","HANK","RUPERT","ROLLAND","KENTON","DAMION","ANTONE","WALDO","FREDRIC","BRADLY","KIP","BURL","WALKER","TYREE","JEFFEREY","AHMED","WILLY","STANFORD","OREN","NOBLE","MOSHE","MIKEL","ENOCH","BRENDON","QUINTIN","JAMISON","FLORENCIO","DARRICK","TOBIAS","HASSAN","GIUSEPPE","DEMARCUS","CLETUS","TYRELL","LYNDON","KEENAN","WERNER","GERALDO","COLUMBUS","CHET","BERTRAM","MARKUS","HUEY","HILTON","DWAIN","DONTE","TYRON","OMER","ISAIAS","HIPOLITO","FERMIN","ADALBERTO","BO","BARRETT","TEODORO","MCKINLEY","MAXIMO","GARFIELD","RALEIGH","LAWERENCE","ABRAM","RASHAD","KING","EMMITT","DARON","SAMUAL","MIQUEL","EUSEBIO","DOMENIC","DARRON","BUSTER","WILBER","RENATO","JC","HOYT","HAYWOOD","EZEKIEL","CHAS","FLORENTINO","ELROY","CLEMENTE","ARDEN","NEVILLE","EDISON","DESHAWN","NATHANIAL","JORDON","DANILO","CLAUD","SHERWOOD","RAYMON","RAYFORD","CRISTOBAL","AMBROSE","TITUS","HYMAN","FELTON","EZEQUIEL","ERASMO","STANTON","LONNY","LEN","IKE","MILAN","LINO","JAROD","HERB","ANDREAS","WALTON","RHETT","PALMER","DOUGLASS","CORDELL","OSWALDO","ELLSWORTH","VIRGILIO","TONEY","NATHANAEL","DEL","BENEDICT","MOSE","JOHNSON","ISREAL","GARRET","FAUSTO","ASA","ARLEN","ZACK","WARNER","MODESTO","FRANCESCO","MANUAL","GAYLORD","GASTON","FILIBERTO","DEANGELO","MICHALE","GRANVILLE","WES","MALIK","ZACKARY","TUAN","ELDRIDGE","CRISTOPHER","CORTEZ","ANTIONE","MALCOM","LONG","KOREY","JOSPEH","COLTON","WAYLON","VON","HOSEA","SHAD","SANTO","RUDOLF","ROLF","REY","RENALDO","MARCELLUS","LUCIUS","KRISTOFER","BOYCE","BENTON","HAYDEN","HARLAND","ARNOLDO","RUEBEN","LEANDRO","KRAIG","JERRELL","JEROMY","HOBERT","CEDRICK","ARLIE","WINFORD","WALLY","LUIGI","KENETH","JACINTO","GRAIG","FRANKLYN","EDMUNDO","SID","PORTER","LEIF","JERAMY","BUCK","WILLIAN","VINCENZO","SHON","LYNWOOD","JERE","HAI","ELDEN","DORSEY","DARELL","BRODERICK","ALONSO"]
	names.sort()
	
	return sum([namevalue(name, names.index(name)) for name in names])

# -----------------------------
# -----------------------------

@SetupAndTime(23,'Sum of all non-abundant numbers:')
def problem23():
	
	def isAbundant(n):

		# Seems like we can filter based on empirical data from prior runs:
		# - Abundant terms must be even or divisible by 5.
		
		if n <= 1: return False
		if n%2 == 1 and n%5 != 0: return False		
		return sum(Prime.factors(n))+1 > n
				
	# "By mathematical analysis, it can be shown that all integers greater than 28123
	# can be written as the sum of two abundant numbers."
	# ^ Good thing to take for granted.
	
	maxterm = 28123
	
	
	z = tuple([n if isAbundant(n) else 0 for n in range(maxterm+1)])
	abundant = tuple([x for x in z if x != 0])
	hashed = set(abundant)
	
	total = 0
	
	for i in xrange(1, maxterm+1):				
		found = False
		for j in abundant:
			if j >= i: break
			if i - j in hashed:
				found = True
				break
				
		if not found:
			total += i
					
	return total
			
# -----------------------------
# -----------------------------

@SetupAndTime(24,'Term %(nthPerm) in permutations of %(digits):', nthPerm=1000000, digits='0123456789')
def problem24(**kwargs):
	nthPerm = kwargs['nthPerm']
	digits = kwargs['digits']
	
	numOn = 0
	
	for term in itertools.permutations(digits):
		numOn += 1
		if numOn == nthPerm:
			return term
			
	return 'There aren\'t that many permutations.'

# -----------------------------
# -----------------------------

@SetupAndTime(25,'First %(numDigits)-digit Fibonacci number:', numDigits=1000)
def problem25(**kwargs):
	numDigits = kwargs['numDigits']
	
	gen = FibGen()
	priorTerm = gen.next()
	
	# Because we start by yielding term 0, we need increase the term counter...
	term = 1
	while len(str(priorTerm)) < numDigits:
		priorTerm = gen.next()
		term += 1
		
	return stringify(str(priorTerm)[0:min(len(str(priorTerm)), 20)], '...', '- term', term)

# -----------------------------
# -----------------------------

@SetupAndTime(26,'Longest fractional reciprocal cycle for 1/1 to 1/%(maxrange)', maxrange=1000)
def problem26(**kwargs):

	def foundTerm(fullString, termToExpand):
		# Expand termToExpand to fit fullString chars.
		expanded = termToExpand*((len(fullString)/len(termToExpand))+1)
		expanded = expanded[0:len(fullString)]
		
		return expanded == fullString

	def getCycle(decstring, i):
		# Kill the integer and decimal point
		string = decstring[decstring.index('.')+1:]
		
		if len(string) <= 1:
			return len(string), string
		
		
		# Just do bruteforce checks.
		maxSlide = (len(string)/2)+1
		
		
		for strmin in xrange(maxSlide):
			for strmax in xrange(strmin+1, maxSlide):
				substr = string[strmin:strmax]
				# If count is ever 1, we can't ever increase it by adding characters.
				if string.count(substr) < 2:
					continue
				else:
					if foundTerm(string[strmin:], substr):
						return len(substr), substr	
						
		return len(string), string
	
	maxrange = kwargs['maxrange']
	maxlen = 0
	termFound = 1
	
	for i in xrange(1,maxrange):
	
		# Because primes (or squares) aren't factors of multiple numbers, they'll have longer
		# fractional periods than strongly composite numbers.
	
		if Prime.isPrime(i) or perfectSquare(i):
		
			term = dec_bits_n(1,i,i*2, True)
			
			l, s = getCycle(term, i)
			
			if l > maxlen:
				maxlen = l
				termFound = i

	return stringify(strfrac(1,termFound), '- length', maxlen)

# -----------------------------
# -----------------------------

@SetupAndTime(27,'For n^2 + a*n + b, A and B generating most primes in (%(minrange),%(maxrange)):', minrange=-1000, maxrange=1000)
def problem27(**kwargs):

	def check(a, b, matches):
		n = 1
		
		# Quick thresholding checks - don't do expensive operations on an entire sequence if we
		# know that the sequence needs to contain a few critical points - namely, at least the end.
		if not Prime.isPrime(pow(matches,2)+(a*matches)+(b)): return 0
		
		while (Prime.isPrime(pow(n,2)+(a*n)+(b))): n += 1
			
		return n
		
		
	largestA, largestB = 0, 0
	mostMatches = 0
	
	minrange = kwargs['minrange']
	maxrange = kwargs['maxrange']
	
	
	# Yeah, this "prune even numbers" assumption doesn't work if we need even numbers in a very limited range, but it will
	# half the search space we need to check as none of the larger results contain even terms for A or B.
	
	minsearch = minrange+1 if minrange%2 == 0 else minrange
	maxsearch = maxrange+1 if maxrange%2 == 1 else maxrange
	
	for a in xrange(minsearch, maxsearch, 2):
		for b in xrange(minsearch, maxsearch, 2):
			matches = check(a, b, mostMatches)
			if matches > mostMatches:
				mostMatches = matches
				largestA = a
				largestB = b
				
	return stringify(mostMatches, 'matches; a=', largestA, 'b=', largestB)
			
# -----------------------------
# -----------------------------

@SetupAndTime(28,'Sum of diagonals in %(spiralBound)x%(spiralBound) grid:', spiralBound=1001)
def problem28(**kwargs):
	spiralBound = kwargs['spiralBound']
	
	terms = [1]
	skip = 2
	spiralOn = 2

	
	while spiralOn < spiralBound and terms[-1] < spiralBound*spiralBound:
		terms.append(terms[-1]+skip)
		terms.append(terms[-1]+skip)
		terms.append(terms[-1]+skip)
		terms.append(terms[-1]+skip)
		
		skip += 2
		spiralOn += 1
	
	return sum(terms)
		
# -----------------------------
# -----------------------------
@SetupAndTime(29,'Distict terms in a^b for a=%(arange), b=%(brange):', arange=[2,100], brange=[2,100])
def problem29(**kwargs):
	
	terms = []
	
	mina, maxa = kwargs['arange']
	minb, maxb = kwargs['brange']
	
	for a in xrange(mina, maxa+1):
		for b in xrange(minb, maxb+1):
			terms.append(pow(a, b))
				
	return len(set(terms))

# -----------------------------
# -----------------------------

@SetupAndTime(30,'Numbers, whose digits to the power of %(maxPower), sum to themselves:', maxPower=5)
def problem30(**kwargs):

	maxPower = kwargs['maxPower']
	
	powerDict = dict()
	for i in range(10):
		powerDict[str(i)] = pow(i, maxPower)
		
	terms = []
	for j in xrange(10**(maxPower-1), 10**maxPower):
		combinedSum = 0
		for term in str(j):
			combinedSum += powerDict[term]
			if combinedSum > j:
				break
		if combinedSum == j:
			terms.append(j)

	return stringify('Terms:', terms, '\nSum:', sum(terms))
		
# -----------------------------
# -----------------------------

@SetupAndTime(31,'Number of ways to make %(amount) from %(coins)', amount=200, coins=[1,2,5,10,20,50,100,200])
def problem31(**kwargs):
	
	amount = kwargs['amount']
	coins = kwargs['coins']

	# Assuming that having no coins is a possibility
	combinations = [1] + [0 for term in range(amount)]
	
	# Coins must be iterated through in increasing order.
	coins.sort()
	
	# Each coin can be added to a combination larger than, and including, itself.
	# This allows all coins to include combinations from other terms summing up to them,
	# but not modify any terms that are smaller.
	for coin in coins:	
		for coinSum in xrange(coin, amount+1):
			
			# The specified combination needs to include additional terms.
			
			# No coins => remainder is zero, which is given to be 1 combination (the coin itself).
			# This solution does require knowing previous state for a lot of terms.
			
			combinations[coinSum] += combinations[coinSum-coin]
		
	return combinations[amount]
		
# -----------------------------
# -----------------------------

@SetupAndTime(32,'A*B=C where A, B, and C are pandigital:')
def problem32(**kwargs):
	
	# At most, len(a) * len(b) + len(a*b) == 9.
	# We can narrow down terms by filtering lengths.
	
	# Doing some math, we know that the multiplied term may have different digits, so....
	
	# a = 1, b = 4, c = 4
	# a = 2, b = 3, c = 4
	
	#Therefore, greatly narrow down our choices and reduce the number of permutations we
	# have to search.
	
	a_lengths = (1, 2)
	b_lengths = (3, 4)
	
	combinations = set()
	
	for a_len in a_lengths:
		for a_permutation in itertools.permutations('123456789',a_len):
			
			# Speedup, if the last digit in a permutation is 1, then it's also an identity and does nothing.
			if a_permutation[-1] == '1': continue
			
			str_a = ''.join(a_permutation)
			int_a = int(str_a)
			
			remaining_permutations = [x for x in '123456789' if x not in a_permutation]
			
			for b_len in b_lengths:
				for b_permutation in itertools.permutations(remaining_permutations, b_len):
					
					# Again, if our permutation ends in 1, it's an identity and will not be valid.
					if b_permutation[-1] == '1': continue
					
					str_b = ''.join(b_permutation)
					int_b = int(str_b)
									
					int_c = int_a*int_b
					str_c = str(int_c)

					# We also don't want a digit 0 in our strings.
					if '0' in str_c: continue
					
					str_ab = str_a + str_b
					possiblyValid = True
					
					for char in str_c:
						if char in str_ab:
							possiblyValid = False
							break
						elif str_c.count(char) != 1:
							possiblyValid = False
							break
							
					if not possiblyValid: continue

					if len(str_c) + len(str_ab) == len('123456789'):
						max_ab = max(int_a, int_b)
						min_ab = min(int_a, int_b)
						
						# Project euler is being silly and having restrictions - "Only include repeated C's once!"
						if int_c not in combinations:
							combinations.add(int_c)
	
	return stringify('\nSum of unique products:', sum([value for value in combinations]))
	
# -----------------------------
# -----------------------------

# Who on earth would think that you could simplify fractions like 49/98 to 4/8 by cancelling the 9's!?
# That's... like saying then that 41/18 = 4/8.  I don't understand the motivation behind this problem.
@SetupAndTime(33,'Digit cancelling fractions')
def problem33(**kwargs):

	# Hardedcoded ways of handling these comparisons, because this problem is weird.
	def hasShare(top, bottom):
		tstr = str(top)
		bstr = str(bottom)
		
		tkeep = [True, True]
		bkeep = [True, True]
		
		if tstr[0] == bstr[0]:
			tkeep[0] = False
			bkeep[0] = False
			
		if tstr[0] == bstr[1]:
			tkeep[0] = False
			bkeep[1] = False
			
		if tstr[1] == bstr[0]:
			tkeep[1] = False
			bkeep[0] = False
			
		if tstr[1] == bstr[1]:
			tkeep[1] = False
			bkeep[1] = False
			
		if not   tkeep[0]: tstr = tstr[1:]
		elif not tkeep[1]: tstr = tstr[0]
		
		if not   bkeep[0]: bstr = bstr[1:]
		elif not bkeep[1]: bstr = bstr[0]
		
		if False in tkeep and tstr != '0' and bstr != '0':
			return (True, int(tstr), int(bstr))
		return (False,)
			
	valid = []
		
	# "There are exactly four non-trivial examples of this type of fraction...
	# "..containing two digits in the numerator and denominator."
	#
	# Okay, so only look for numerators between 10 and 100, and denoms between numerator and 100.
	
	for top in xrange(10, 100):
		for bottom in xrange(top+1, 100):
		
			value = hasShare(top, bottom)
			
			if value[0]:
			
				# Arbitrary precision, woo...
				if dec_bits_n(top, bottom, 100) == dec_bits_n(value[1], value[2], 100):
					if top % 10 != 0:
						print str(top) + '/' + str(bottom), '->', strfrac(value[1], value[2])
						valid.append((value[1], value[2]))
						
	top =    mul([val[0] for val in valid])
	bottom = mul([val[1] for val in valid])
	
	divisor = Prime.gcd(top, bottom)
	
	return stringify(strfrac(top, bottom), '-> GCD:', divisor, '->', strfrac(top/divisor, bottom/divisor))

# -----------------------------
# -----------------------------

@SetupAndTime(34,'All terms where sum of factorial digits == sum of term:')
def problem34(**kwargs):
	

	def facdigits(n, lookupTable):
		return sum([lookupTable[digit] for digit in str(n)]) == n
		
	def fast_facdigits(n, lookupTable):
		j = 0
		
		for digit in str(n):
			j += lookupTable[digit]
			if j > n: return False
			
		return j==n
			
	
	start = 3
	
	table = dict()
	for value in '0123456789':
		table[value] = math.factorial(int(value))

	values = []	
	
	# Implication of the question is that there has to be a point where no other factorial digit thing occurs.	
	# So...at what value does 9! + 9!..... + 9! < 9999...99
	#
	# Programmatically generating this yields 2540160.  Rather than always generate it, let's make it a constant.
	
	maxFactoralDigitMatch = 2540160
		
	for i in xrange(start, maxFactoralDigitMatch+1):
		if fast_facdigits(i, table):
			values.append(i)
			
	return stringify(values, '(sum:', str(sum(values)) + ')')


# -----------------------------
# -----------------------------

@SetupAndTime(35,'Number of circular primes below %(maxTerm):', maxTerm=1000000)
def problem35(**kwargs):
	
	def CircularPrime(term):	
		# Perhaps another optimization.  Don't do Prime.isPrime on every permutation.
		# Instead, look at one term at a time and try to break early if possible.
		
		terms = []
			
		for c in rotations(term):
			intval = int(''.join(c))
			if not Prime.isPrime(intval): return []
			terms.append(intval)

		return terms
	
	def noBadEvenCircles(endAt):
		
		for j in ['2', '3', '5', '7']:
			yield j
			
		# Introduce tons of cyclomatic complexity to make things faster!
		# Because we can't have evens anywhere (divis by 2) or 5 anywhere (divis by 5),
		# we can greatly reduce the search space we need to look at.
		term = '1379'
		
		for size in xrange(2, len(str(endAt))):
		
			usedTerms = set()
		
			for combination in itertools.product(term, repeat=size):
				
				strval = ''.join(combination)
				if strval in usedTerms: continue

				if Prime.isPrime(int(strval)): yield strval
					
				usedTerms.update([p for p in rotations(strval)])
				
	maxTerm = kwargs['maxTerm']
	terms = set()
	for i in noBadEvenCircles(maxTerm):	terms.update(CircularPrime(i))
	
	return stringify(len(terms), '-', sorted(terms))

# -----------------------------
# -----------------------------

@SetupAndTime(36,'Numbers below %(maxTerm) palindromic in base 10 and base 2:', maxTerm=1000000)
def problem36(**kwargs):
	maxTerm = kwargs['maxTerm']
	
	total = 0
	
	for i in xrange(10, maxTerm+1):
		if isPalindrome(str(i)) and isPalindrome(bin(i)[2:]):
			total += i
			
	return total
	
# -----------------------------
# -----------------------------

@SetupAndTime(37,'Sum of all mirrored, truncatable primes:')
def problem37(**kwargs):

	def isMirrorTruncatePrime(number):
	
		if not Prime.isPrime(number): return False
		if number < 10: return False

		forward = number/10
		backwardStr = str(number)[1:]
		
		while forward != 0:
			if not Prime.isPrime(forward): return False
			forward /= 10
		
		while backwardStr != '':
			if not Prime.isPrime(int(backwardStr)): return False
			backwardStr = backwardStr[1:]
		
		return True
		
	
	counter = 0
	total = 0
	
	for prime in Prime.ordered:
		if counter == 11: return total
		if isMirrorTruncatePrime(prime): 
			total += prime
			counter += 1

# -----------------------------
# -----------------------------

@SetupAndTime(38,'The largest pandigital chain made from factors and a product:')
def problem38(**kwargs):
	
	def pandaformat(number):
		
		string = ''
		n = 0
		invalid = False
		
		while not invalid:
			n += 1
			string += str(n * number)
			if len(string) >= 9:
				if len(string) > 9:
					invalid = True
				break
			
	
		if not invalid and n > 1 and stringPermutation(string, '123456789'): return [n, string]
			
		else: return None
		
	largest = '1'
	bign = 1
			
	# Despite the max nine digit term being '987654321', we know that n must be >= 2.
	# This means that the max value of the range(0, n) must be at most floor(9/2) chars long.
	
	maxRange = int('1'*int(9/2))*9
	
	for i in xrange(maxRange):
			
		valid = pandaformat(i)
		if valid:
			if valid[0] > 1 and valid[1] > largest:
				largest = valid[1]
				bign = valid[0]
				print largest, ', n =', bign, ', i =', i
			
	return '\n'+largest
		
# -----------------------------
# -----------------------------
				
@SetupAndTime(39,'Value for p<=%(maxP) that creates the most right triangles:', maxP=1000)
def problem39(**kwargs):
	# Create a list of all possible a^2 + b^2 == c^2 for a, b, c <= 1000?
	maxP = kwargs['maxP']
	
	combinations = []
	
	# We know the triangle can't be equilateral, so only go down a third of the total thing?
	
	lengths = dict()
		
	for x in xrange(maxP-2, 0, -1):		

		# Y must be <= X.
		maxY = min(x, maxP - x - 1)
		
		for y in xrange(maxY, 0, -1):
		
			# sqrtxy must be <= Y to prevent searching repeated terms between sqrtxy and y.
			sqrtxy = int(math.sqrt(x**2 - y**2))
			
			if x+y+sqrtxy > maxP or sqrtxy > y or sqrtxy == 0: continue
			
			if sqrtxy**2 + y**2 == x**2:
				z = sqrtxy + y + x
				if not z in lengths: lengths[z] = set()
				lengths[z].add((sqrtxy, y, x))
	
	best = 0
	for g in lengths:
		if best not in lengths:
			best = g
		elif len(lengths[g]) > len(lengths[best]):
			best = g
		
	return stringify(best, 'has', len(lengths[best]), 'terms -', lengths[best])
	

# -----------------------------
# -----------------------------		

@SetupAndTime(40,'Product of digits at indices %(indices) in the term 0.1234567891011...', indices=[1,10,100,1000,10000,100000,1000000])
def problem40(**kwargs):
	def convertPosition(index):

		digitsUsed = 0
		power = 0
		
		tenpower = 10**(power)
		while index > tenpower:
		
			notIncluded = tenpower/10 if power > 1 else 0
			usableDigits = tenpower - notIncluded
			
			digitsUsed += usableDigits * power	
			
			power += 1
			tenpower = 10**(power)
			
		adjustedIndex = index - (tenpower/10 if power > 1 else 0)
	
		return digitsUsed + power*adjustedIndex
			
	def tellPosition(location):
				
		n = 2
		pow = 0
		
		while convertPosition(n**(pow+1)) < location:
			pow += 1
		
		lower = n**pow
		higher = lower*n
		
		numberOn, numSubstrPos = 0, 0
		
		if location <= 9: return location
		
		while 1:
			
			# Being silly and doing adaptive guessing on the bounds.
			
			avg = (lower + higher)/2
			converted = convertPosition(avg)
			convertedPlusOne = convertPosition(avg+1)
			
			if converted <= location and convertedPlusOne > location:
				numberOn = avg
				numSubstrPos = (location - converted) % (convertedPlusOne - converted)
				break
				
			elif converted > location:                                  higher = avg
			elif convertedPlusOne < location:                           lower  = avg+1
			elif converted < location and convertedPlusOne >= location: lower  = avg+1

		#print 'Substr Position=', location, 'Number Represented=', numberOn, 'Digit at idx', numSubstrPos, ':', str(numberOn)[numSubstrPos]
		return str(numberOn)[numSubstrPos]
	
	
	terms = kwargs['indices']
	values = [int(tellPosition(term)) for term in terms]
	return stringify(values, '->', mul(values))
		
# -----------------------------
# -----------------------------

@SetupAndTime(41,'Largest prime containing all digits 1 to 9:')
def problem41(**kwargs):

	digits = '123456789'
	primeEnds = '13579'	
	
	largestPrime = None
	
	while len(digits) > 1:		
	
		# If the sum of the digits is divisible by 3, the number is divisible by 3 --- and thus not prime.
		if sum([int(x) for x in digits]) % 3 != 0:
		
			# Do permutations through the digits - but don't permute the whole string.
			# We know the string has to end in an odd number to be prime, so we can
			# half the search space by only allowing for odd endings to be appended
			# separately.
			for iter in itertools.permutations(digits[::-1], len(digits)-1):
			
				strIter = ''.join(iter)
				
				# Only append the ending if it's not already in the string.
				endings = [end for end in primeEnds if end not in strIter]
				
				for end in endings:
				
					tmpIter = strIter + end
					numericIter = int(tmpIter)
					
					if Prime.isPrime(numericIter):
						return tmpIter
		
		digits = digits[:-1]
		primeEnds = [x for x in primeEnds if x in digits]

	return 'No primes fit the criteria...'
			
# -----------------------------
# -----------------------------

@SetupAndTime(42,'Total number of triangle words:')
def problem42(**kwargs):	
	
	strings = ["A","ABILITY","ABLE","ABOUT","ABOVE","ABSENCE","ABSOLUTELY","ACADEMIC","ACCEPT","ACCESS","ACCIDENT","ACCOMPANY","ACCORDING","ACCOUNT","ACHIEVE","ACHIEVEMENT","ACID","ACQUIRE","ACROSS","ACT","ACTION","ACTIVE","ACTIVITY","ACTUAL","ACTUALLY","ADD","ADDITION","ADDITIONAL","ADDRESS","ADMINISTRATION","ADMIT","ADOPT","ADULT","ADVANCE","ADVANTAGE","ADVICE","ADVISE","AFFAIR","AFFECT","AFFORD","AFRAID","AFTER","AFTERNOON","AFTERWARDS","AGAIN","AGAINST","AGE","AGENCY","AGENT","AGO","AGREE","AGREEMENT","AHEAD","AID","AIM","AIR","AIRCRAFT","ALL","ALLOW","ALMOST","ALONE","ALONG","ALREADY","ALRIGHT","ALSO","ALTERNATIVE","ALTHOUGH","ALWAYS","AMONG","AMONGST","AMOUNT","AN","ANALYSIS","ANCIENT","AND","ANIMAL","ANNOUNCE","ANNUAL","ANOTHER","ANSWER","ANY","ANYBODY","ANYONE","ANYTHING","ANYWAY","APART","APPARENT","APPARENTLY","APPEAL","APPEAR","APPEARANCE","APPLICATION","APPLY","APPOINT","APPOINTMENT","APPROACH","APPROPRIATE","APPROVE","AREA","ARGUE","ARGUMENT","ARISE","ARM","ARMY","AROUND","ARRANGE","ARRANGEMENT","ARRIVE","ART","ARTICLE","ARTIST","AS","ASK","ASPECT","ASSEMBLY","ASSESS","ASSESSMENT","ASSET","ASSOCIATE","ASSOCIATION","ASSUME","ASSUMPTION","AT","ATMOSPHERE","ATTACH","ATTACK","ATTEMPT","ATTEND","ATTENTION","ATTITUDE","ATTRACT","ATTRACTIVE","AUDIENCE","AUTHOR","AUTHORITY","AVAILABLE","AVERAGE","AVOID","AWARD","AWARE","AWAY","AYE","BABY","BACK","BACKGROUND","BAD","BAG","BALANCE","BALL","BAND","BANK","BAR","BASE","BASIC","BASIS","BATTLE","BE","BEAR","BEAT","BEAUTIFUL","BECAUSE","BECOME","BED","BEDROOM","BEFORE","BEGIN","BEGINNING","BEHAVIOUR","BEHIND","BELIEF","BELIEVE","BELONG","BELOW","BENEATH","BENEFIT","BESIDE","BEST","BETTER","BETWEEN","BEYOND","BIG","BILL","BIND","BIRD","BIRTH","BIT","BLACK","BLOCK","BLOOD","BLOODY","BLOW","BLUE","BOARD","BOAT","BODY","BONE","BOOK","BORDER","BOTH","BOTTLE","BOTTOM","BOX","BOY","BRAIN","BRANCH","BREAK","BREATH","BRIDGE","BRIEF","BRIGHT","BRING","BROAD","BROTHER","BUDGET","BUILD","BUILDING","BURN","BUS","BUSINESS","BUSY","BUT","BUY","BY","CABINET","CALL","CAMPAIGN","CAN","CANDIDATE","CAPABLE","CAPACITY","CAPITAL","CAR","CARD","CARE","CAREER","CAREFUL","CAREFULLY","CARRY","CASE","CASH","CAT","CATCH","CATEGORY","CAUSE","CELL","CENTRAL","CENTRE","CENTURY","CERTAIN","CERTAINLY","CHAIN","CHAIR","CHAIRMAN","CHALLENGE","CHANCE","CHANGE","CHANNEL","CHAPTER","CHARACTER","CHARACTERISTIC","CHARGE","CHEAP","CHECK","CHEMICAL","CHIEF","CHILD","CHOICE","CHOOSE","CHURCH","CIRCLE","CIRCUMSTANCE","CITIZEN","CITY","CIVIL","CLAIM","CLASS","CLEAN","CLEAR","CLEARLY","CLIENT","CLIMB","CLOSE","CLOSELY","CLOTHES","CLUB","COAL","CODE","COFFEE","COLD","COLLEAGUE","COLLECT","COLLECTION","COLLEGE","COLOUR","COMBINATION","COMBINE","COME","COMMENT","COMMERCIAL","COMMISSION","COMMIT","COMMITMENT","COMMITTEE","COMMON","COMMUNICATION","COMMUNITY","COMPANY","COMPARE","COMPARISON","COMPETITION","COMPLETE","COMPLETELY","COMPLEX","COMPONENT","COMPUTER","CONCENTRATE","CONCENTRATION","CONCEPT","CONCERN","CONCERNED","CONCLUDE","CONCLUSION","CONDITION","CONDUCT","CONFERENCE","CONFIDENCE","CONFIRM","CONFLICT","CONGRESS","CONNECT","CONNECTION","CONSEQUENCE","CONSERVATIVE","CONSIDER","CONSIDERABLE","CONSIDERATION","CONSIST","CONSTANT","CONSTRUCTION","CONSUMER","CONTACT","CONTAIN","CONTENT","CONTEXT","CONTINUE","CONTRACT","CONTRAST","CONTRIBUTE","CONTRIBUTION","CONTROL","CONVENTION","CONVERSATION","COPY","CORNER","CORPORATE","CORRECT","COS","COST","COULD","COUNCIL","COUNT","COUNTRY","COUNTY","COUPLE","COURSE","COURT","COVER","CREATE","CREATION","CREDIT","CRIME","CRIMINAL","CRISIS","CRITERION","CRITICAL","CRITICISM","CROSS","CROWD","CRY","CULTURAL","CULTURE","CUP","CURRENT","CURRENTLY","CURRICULUM","CUSTOMER","CUT","DAMAGE","DANGER","DANGEROUS","DARK","DATA","DATE","DAUGHTER","DAY","DEAD","DEAL","DEATH","DEBATE","DEBT","DECADE","DECIDE","DECISION","DECLARE","DEEP","DEFENCE","DEFENDANT","DEFINE","DEFINITION","DEGREE","DELIVER","DEMAND","DEMOCRATIC","DEMONSTRATE","DENY","DEPARTMENT","DEPEND","DEPUTY","DERIVE","DESCRIBE","DESCRIPTION","DESIGN","DESIRE","DESK","DESPITE","DESTROY","DETAIL","DETAILED","DETERMINE","DEVELOP","DEVELOPMENT","DEVICE","DIE","DIFFERENCE","DIFFERENT","DIFFICULT","DIFFICULTY","DINNER","DIRECT","DIRECTION","DIRECTLY","DIRECTOR","DISAPPEAR","DISCIPLINE","DISCOVER","DISCUSS","DISCUSSION","DISEASE","DISPLAY","DISTANCE","DISTINCTION","DISTRIBUTION","DISTRICT","DIVIDE","DIVISION","DO","DOCTOR","DOCUMENT","DOG","DOMESTIC","DOOR","DOUBLE","DOUBT","DOWN","DRAW","DRAWING","DREAM","DRESS","DRINK","DRIVE","DRIVER","DROP","DRUG","DRY","DUE","DURING","DUTY","EACH","EAR","EARLY","EARN","EARTH","EASILY","EAST","EASY","EAT","ECONOMIC","ECONOMY","EDGE","EDITOR","EDUCATION","EDUCATIONAL","EFFECT","EFFECTIVE","EFFECTIVELY","EFFORT","EGG","EITHER","ELDERLY","ELECTION","ELEMENT","ELSE","ELSEWHERE","EMERGE","EMPHASIS","EMPLOY","EMPLOYEE","EMPLOYER","EMPLOYMENT","EMPTY","ENABLE","ENCOURAGE","END","ENEMY","ENERGY","ENGINE","ENGINEERING","ENJOY","ENOUGH","ENSURE","ENTER","ENTERPRISE","ENTIRE","ENTIRELY","ENTITLE","ENTRY","ENVIRONMENT","ENVIRONMENTAL","EQUAL","EQUALLY","EQUIPMENT","ERROR","ESCAPE","ESPECIALLY","ESSENTIAL","ESTABLISH","ESTABLISHMENT","ESTATE","ESTIMATE","EVEN","EVENING","EVENT","EVENTUALLY","EVER","EVERY","EVERYBODY","EVERYONE","EVERYTHING","EVIDENCE","EXACTLY","EXAMINATION","EXAMINE","EXAMPLE","EXCELLENT","EXCEPT","EXCHANGE","EXECUTIVE","EXERCISE","EXHIBITION","EXIST","EXISTENCE","EXISTING","EXPECT","EXPECTATION","EXPENDITURE","EXPENSE","EXPENSIVE","EXPERIENCE","EXPERIMENT","EXPERT","EXPLAIN","EXPLANATION","EXPLORE","EXPRESS","EXPRESSION","EXTEND","EXTENT","EXTERNAL","EXTRA","EXTREMELY","EYE","FACE","FACILITY","FACT","FACTOR","FACTORY","FAIL","FAILURE","FAIR","FAIRLY","FAITH","FALL","FAMILIAR","FAMILY","FAMOUS","FAR","FARM","FARMER","FASHION","FAST","FATHER","FAVOUR","FEAR","FEATURE","FEE","FEEL","FEELING","FEMALE","FEW","FIELD","FIGHT","FIGURE","FILE","FILL","FILM","FINAL","FINALLY","FINANCE","FINANCIAL","FIND","FINDING","FINE","FINGER","FINISH","FIRE","FIRM","FIRST","FISH","FIT","FIX","FLAT","FLIGHT","FLOOR","FLOW","FLOWER","FLY","FOCUS","FOLLOW","FOLLOWING","FOOD","FOOT","FOOTBALL","FOR","FORCE","FOREIGN","FOREST","FORGET","FORM","FORMAL","FORMER","FORWARD","FOUNDATION","FREE","FREEDOM","FREQUENTLY","FRESH","FRIEND","FROM","FRONT","FRUIT","FUEL","FULL","FULLY","FUNCTION","FUND","FUNNY","FURTHER","FUTURE","GAIN","GAME","GARDEN","GAS","GATE","GATHER","GENERAL","GENERALLY","GENERATE","GENERATION","GENTLEMAN","GET","GIRL","GIVE","GLASS","GO","GOAL","GOD","GOLD","GOOD","GOVERNMENT","GRANT","GREAT","GREEN","GREY","GROUND","GROUP","GROW","GROWING","GROWTH","GUEST","GUIDE","GUN","HAIR","HALF","HALL","HAND","HANDLE","HANG","HAPPEN","HAPPY","HARD","HARDLY","HATE","HAVE","HE","HEAD","HEALTH","HEAR","HEART","HEAT","HEAVY","HELL","HELP","HENCE","HER","HERE","HERSELF","HIDE","HIGH","HIGHLY","HILL","HIM","HIMSELF","HIS","HISTORICAL","HISTORY","HIT","HOLD","HOLE","HOLIDAY","HOME","HOPE","HORSE","HOSPITAL","HOT","HOTEL","HOUR","HOUSE","HOUSEHOLD","HOUSING","HOW","HOWEVER","HUGE","HUMAN","HURT","HUSBAND","I","IDEA","IDENTIFY","IF","IGNORE","ILLUSTRATE","IMAGE","IMAGINE","IMMEDIATE","IMMEDIATELY","IMPACT","IMPLICATION","IMPLY","IMPORTANCE","IMPORTANT","IMPOSE","IMPOSSIBLE","IMPRESSION","IMPROVE","IMPROVEMENT","IN","INCIDENT","INCLUDE","INCLUDING","INCOME","INCREASE","INCREASED","INCREASINGLY","INDEED","INDEPENDENT","INDEX","INDICATE","INDIVIDUAL","INDUSTRIAL","INDUSTRY","INFLUENCE","INFORM","INFORMATION","INITIAL","INITIATIVE","INJURY","INSIDE","INSIST","INSTANCE","INSTEAD","INSTITUTE","INSTITUTION","INSTRUCTION","INSTRUMENT","INSURANCE","INTEND","INTENTION","INTEREST","INTERESTED","INTERESTING","INTERNAL","INTERNATIONAL","INTERPRETATION","INTERVIEW","INTO","INTRODUCE","INTRODUCTION","INVESTIGATE","INVESTIGATION","INVESTMENT","INVITE","INVOLVE","IRON","IS","ISLAND","ISSUE","IT","ITEM","ITS","ITSELF","JOB","JOIN","JOINT","JOURNEY","JUDGE","JUMP","JUST","JUSTICE","KEEP","KEY","KID","KILL","KIND","KING","KITCHEN","KNEE","KNOW","KNOWLEDGE","LABOUR","LACK","LADY","LAND","LANGUAGE","LARGE","LARGELY","LAST","LATE","LATER","LATTER","LAUGH","LAUNCH","LAW","LAWYER","LAY","LEAD","LEADER","LEADERSHIP","LEADING","LEAF","LEAGUE","LEAN","LEARN","LEAST","LEAVE","LEFT","LEG","LEGAL","LEGISLATION","LENGTH","LESS","LET","LETTER","LEVEL","LIABILITY","LIBERAL","LIBRARY","LIE","LIFE","LIFT","LIGHT","LIKE","LIKELY","LIMIT","LIMITED","LINE","LINK","LIP","LIST","LISTEN","LITERATURE","LITTLE","LIVE","LIVING","LOAN","LOCAL","LOCATION","LONG","LOOK","LORD","LOSE","LOSS","LOT","LOVE","LOVELY","LOW","LUNCH","MACHINE","MAGAZINE","MAIN","MAINLY","MAINTAIN","MAJOR","MAJORITY","MAKE","MALE","MAN","MANAGE","MANAGEMENT","MANAGER","MANNER","MANY","MAP","MARK","MARKET","MARRIAGE","MARRIED","MARRY","MASS","MASTER","MATCH","MATERIAL","MATTER","MAY","MAYBE","ME","MEAL","MEAN","MEANING","MEANS","MEANWHILE","MEASURE","MECHANISM","MEDIA","MEDICAL","MEET","MEETING","MEMBER","MEMBERSHIP","MEMORY","MENTAL","MENTION","MERELY","MESSAGE","METAL","METHOD","MIDDLE","MIGHT","MILE","MILITARY","MILK","MIND","MINE","MINISTER","MINISTRY","MINUTE","MISS","MISTAKE","MODEL","MODERN","MODULE","MOMENT","MONEY","MONTH","MORE","MORNING","MOST","MOTHER","MOTION","MOTOR","MOUNTAIN","MOUTH","MOVE","MOVEMENT","MUCH","MURDER","MUSEUM","MUSIC","MUST","MY","MYSELF","NAME","NARROW","NATION","NATIONAL","NATURAL","NATURE","NEAR","NEARLY","NECESSARILY","NECESSARY","NECK","NEED","NEGOTIATION","NEIGHBOUR","NEITHER","NETWORK","NEVER","NEVERTHELESS","NEW","NEWS","NEWSPAPER","NEXT","NICE","NIGHT","NO","NOBODY","NOD","NOISE","NONE","NOR","NORMAL","NORMALLY","NORTH","NORTHERN","NOSE","NOT","NOTE","NOTHING","NOTICE","NOTION","NOW","NUCLEAR","NUMBER","NURSE","OBJECT","OBJECTIVE","OBSERVATION","OBSERVE","OBTAIN","OBVIOUS","OBVIOUSLY","OCCASION","OCCUR","ODD","OF","OFF","OFFENCE","OFFER","OFFICE","OFFICER","OFFICIAL","OFTEN","OIL","OKAY","OLD","ON","ONCE","ONE","ONLY","ONTO","OPEN","OPERATE","OPERATION","OPINION","OPPORTUNITY","OPPOSITION","OPTION","OR","ORDER","ORDINARY","ORGANISATION","ORGANISE","ORGANIZATION","ORIGIN","ORIGINAL","OTHER","OTHERWISE","OUGHT","OUR","OURSELVES","OUT","OUTCOME","OUTPUT","OUTSIDE","OVER","OVERALL","OWN","OWNER","PACKAGE","PAGE","PAIN","PAINT","PAINTING","PAIR","PANEL","PAPER","PARENT","PARK","PARLIAMENT","PART","PARTICULAR","PARTICULARLY","PARTLY","PARTNER","PARTY","PASS","PASSAGE","PAST","PATH","PATIENT","PATTERN","PAY","PAYMENT","PEACE","PENSION","PEOPLE","PER","PERCENT","PERFECT","PERFORM","PERFORMANCE","PERHAPS","PERIOD","PERMANENT","PERSON","PERSONAL","PERSUADE","PHASE","PHONE","PHOTOGRAPH","PHYSICAL","PICK","PICTURE","PIECE","PLACE","PLAN","PLANNING","PLANT","PLASTIC","PLATE","PLAY","PLAYER","PLEASE","PLEASURE","PLENTY","PLUS","POCKET","POINT","POLICE","POLICY","POLITICAL","POLITICS","POOL","POOR","POPULAR","POPULATION","POSITION","POSITIVE","POSSIBILITY","POSSIBLE","POSSIBLY","POST","POTENTIAL","POUND","POWER","POWERFUL","PRACTICAL","PRACTICE","PREFER","PREPARE","PRESENCE","PRESENT","PRESIDENT","PRESS","PRESSURE","PRETTY","PREVENT","PREVIOUS","PREVIOUSLY","PRICE","PRIMARY","PRIME","PRINCIPLE","PRIORITY","PRISON","PRISONER","PRIVATE","PROBABLY","PROBLEM","PROCEDURE","PROCESS","PRODUCE","PRODUCT","PRODUCTION","PROFESSIONAL","PROFIT","PROGRAM","PROGRAMME","PROGRESS","PROJECT","PROMISE","PROMOTE","PROPER","PROPERLY","PROPERTY","PROPORTION","PROPOSE","PROPOSAL","PROSPECT","PROTECT","PROTECTION","PROVE","PROVIDE","PROVIDED","PROVISION","PUB","PUBLIC","PUBLICATION","PUBLISH","PULL","PUPIL","PURPOSE","PUSH","PUT","QUALITY","QUARTER","QUESTION","QUICK","QUICKLY","QUIET","QUITE","RACE","RADIO","RAILWAY","RAIN","RAISE","RANGE","RAPIDLY","RARE","RATE","RATHER","REACH","REACTION","READ","READER","READING","READY","REAL","REALISE","REALITY","REALIZE","REALLY","REASON","REASONABLE","RECALL","RECEIVE","RECENT","RECENTLY","RECOGNISE","RECOGNITION","RECOGNIZE","RECOMMEND","RECORD","RECOVER","RED","REDUCE","REDUCTION","REFER","REFERENCE","REFLECT","REFORM","REFUSE","REGARD","REGION","REGIONAL","REGULAR","REGULATION","REJECT","RELATE","RELATION","RELATIONSHIP","RELATIVE","RELATIVELY","RELEASE","RELEVANT","RELIEF","RELIGION","RELIGIOUS","RELY","REMAIN","REMEMBER","REMIND","REMOVE","REPEAT","REPLACE","REPLY","REPORT","REPRESENT","REPRESENTATION","REPRESENTATIVE","REQUEST","REQUIRE","REQUIREMENT","RESEARCH","RESOURCE","RESPECT","RESPOND","RESPONSE","RESPONSIBILITY","RESPONSIBLE","REST","RESTAURANT","RESULT","RETAIN","RETURN","REVEAL","REVENUE","REVIEW","REVOLUTION","RICH","RIDE","RIGHT","RING","RISE","RISK","RIVER","ROAD","ROCK","ROLE","ROLL","ROOF","ROOM","ROUND","ROUTE","ROW","ROYAL","RULE","RUN","RURAL","SAFE","SAFETY","SALE","SAME","SAMPLE","SATISFY","SAVE","SAY","SCALE","SCENE","SCHEME","SCHOOL","SCIENCE","SCIENTIFIC","SCIENTIST","SCORE","SCREEN","SEA","SEARCH","SEASON","SEAT","SECOND","SECONDARY","SECRETARY","SECTION","SECTOR","SECURE","SECURITY","SEE","SEEK","SEEM","SELECT","SELECTION","SELL","SEND","SENIOR","SENSE","SENTENCE","SEPARATE","SEQUENCE","SERIES","SERIOUS","SERIOUSLY","SERVANT","SERVE","SERVICE","SESSION","SET","SETTLE","SETTLEMENT","SEVERAL","SEVERE","SEX","SEXUAL","SHAKE","SHALL","SHAPE","SHARE","SHE","SHEET","SHIP","SHOE","SHOOT","SHOP","SHORT","SHOT","SHOULD","SHOULDER","SHOUT","SHOW","SHUT","SIDE","SIGHT","SIGN","SIGNAL","SIGNIFICANCE","SIGNIFICANT","SILENCE","SIMILAR","SIMPLE","SIMPLY","SINCE","SING","SINGLE","SIR","SISTER","SIT","SITE","SITUATION","SIZE","SKILL","SKIN","SKY","SLEEP","SLIGHTLY","SLIP","SLOW","SLOWLY","SMALL","SMILE","SO","SOCIAL","SOCIETY","SOFT","SOFTWARE","SOIL","SOLDIER","SOLICITOR","SOLUTION","SOME","SOMEBODY","SOMEONE","SOMETHING","SOMETIMES","SOMEWHAT","SOMEWHERE","SON","SONG","SOON","SORRY","SORT","SOUND","SOURCE","SOUTH","SOUTHERN","SPACE","SPEAK","SPEAKER","SPECIAL","SPECIES","SPECIFIC","SPEECH","SPEED","SPEND","SPIRIT","SPORT","SPOT","SPREAD","SPRING","STAFF","STAGE","STAND","STANDARD","STAR","START","STATE","STATEMENT","STATION","STATUS","STAY","STEAL","STEP","STICK","STILL","STOCK","STONE","STOP","STORE","STORY","STRAIGHT","STRANGE","STRATEGY","STREET","STRENGTH","STRIKE","STRONG","STRONGLY","STRUCTURE","STUDENT","STUDIO","STUDY","STUFF","STYLE","SUBJECT","SUBSTANTIAL","SUCCEED","SUCCESS","SUCCESSFUL","SUCH","SUDDENLY","SUFFER","SUFFICIENT","SUGGEST","SUGGESTION","SUITABLE","SUM","SUMMER","SUN","SUPPLY","SUPPORT","SUPPOSE","SURE","SURELY","SURFACE","SURPRISE","SURROUND","SURVEY","SURVIVE","SWITCH","SYSTEM","TABLE","TAKE","TALK","TALL","TAPE","TARGET","TASK","TAX","TEA","TEACH","TEACHER","TEACHING","TEAM","TEAR","TECHNICAL","TECHNIQUE","TECHNOLOGY","TELEPHONE","TELEVISION","TELL","TEMPERATURE","TEND","TERM","TERMS","TERRIBLE","TEST","TEXT","THAN","THANK","THANKS","THAT","THE","THEATRE","THEIR","THEM","THEME","THEMSELVES","THEN","THEORY","THERE","THEREFORE","THESE","THEY","THIN","THING","THINK","THIS","THOSE","THOUGH","THOUGHT","THREAT","THREATEN","THROUGH","THROUGHOUT","THROW","THUS","TICKET","TIME","TINY","TITLE","TO","TODAY","TOGETHER","TOMORROW","TONE","TONIGHT","TOO","TOOL","TOOTH","TOP","TOTAL","TOTALLY","TOUCH","TOUR","TOWARDS","TOWN","TRACK","TRADE","TRADITION","TRADITIONAL","TRAFFIC","TRAIN","TRAINING","TRANSFER","TRANSPORT","TRAVEL","TREAT","TREATMENT","TREATY","TREE","TREND","TRIAL","TRIP","TROOP","TROUBLE","TRUE","TRUST","TRUTH","TRY","TURN","TWICE","TYPE","TYPICAL","UNABLE","UNDER","UNDERSTAND","UNDERSTANDING","UNDERTAKE","UNEMPLOYMENT","UNFORTUNATELY","UNION","UNIT","UNITED","UNIVERSITY","UNLESS","UNLIKELY","UNTIL","UP","UPON","UPPER","URBAN","US","USE","USED","USEFUL","USER","USUAL","USUALLY","VALUE","VARIATION","VARIETY","VARIOUS","VARY","VAST","VEHICLE","VERSION","VERY","VIA","VICTIM","VICTORY","VIDEO","VIEW","VILLAGE","VIOLENCE","VISION","VISIT","VISITOR","VITAL","VOICE","VOLUME","VOTE","WAGE","WAIT","WALK","WALL","WANT","WAR","WARM","WARN","WASH","WATCH","WATER","WAVE","WAY","WE","WEAK","WEAPON","WEAR","WEATHER","WEEK","WEEKEND","WEIGHT","WELCOME","WELFARE","WELL","WEST","WESTERN","WHAT","WHATEVER","WHEN","WHERE","WHEREAS","WHETHER","WHICH","WHILE","WHILST","WHITE","WHO","WHOLE","WHOM","WHOSE","WHY","WIDE","WIDELY","WIFE","WILD","WILL","WIN","WIND","WINDOW","WINE","WING","WINNER","WINTER","WISH","WITH","WITHDRAW","WITHIN","WITHOUT","WOMAN","WONDER","WONDERFUL","WOOD","WORD","WORK","WORKER","WORKING","WORKS","WORLD","WORRY","WORTH","WOULD","WRITE","WRITER","WRITING","WRONG","YARD","YEAH","YEAR","YES","YESTERDAY","YET","YOU","YOUNG","YOUR","YOURSELF","YOUTH"]
	
	def wordValue(word):
	
		# Unnecessary ASCII sanitization
		upperword = word.upper()
		return sum([ord(char) - ord('A') + 1 for char in upperword])
		
	
	valdict = dict()
	largestValue = 0
	
	for word in strings:
	
		value = wordValue(word)
		
		if value > largestValue:
			largestValue = value
			
		if value in valdict:
			valdict[value].append(word)
			
		else:
			valdict[value] = [word]
			
	triangleTerms = [1]
	i = 2
	
	while triangleTerms[-1] < largestValue:
		triangleTerms.append((i*(i+1))/2)
		i += 1
	
	totalwords = 0
	
	for term in triangleTerms:
		if term in valdict:
			totalwords += len(valdict[term])
			
	return totalwords
			

# -----------------------------
# -----------------------------

@SetupAndTime(43,'Pandigital numbers with prime substring divisibility:')
def problem43(**kwargs):
	
	def noRepeats(ending):
		for char in ending:
			if ending.count(char) > 1: return False
		return True
	
	def remaining(ending):
		return ''.join([missing for missing in numericConstructs if not (missing in ending)])
		
	numericConstructs = '1234567890'

	# Construct a shared digit list to make filtering much faster		
	endingDivisors = [7, 11, 13, 17]
	maxes = [(1000/i)+1 for i in endingDivisors]
	
	terms = []
	for divisor, max in zip(endingDivisors, maxes):
		terms.append([str(a*divisor).zfill(3) for a in xrange(1, max)])
	
	endings = terms[0]
	for term in terms[1:]:
		both = []
		for a in endings:
			for b in term:
				if a[-2:] == b[:2]:
					both.append(a+b[2])
		endings = both

	# We know the term itself cannot duplicate digits.
	# We can prume the list somewhat for itertools by limiting the number of permutations
	# it needs to make per each individual ending, rather than just matching a valid ending.
	endings = [(ending, remaining(ending)) for ending in endings if noRepeats(ending)]
	
	goodterms = []

	for base in endings:
	
		ending, construct = base[0], base[1]
		
		for iter in itertools.permutations(construct):
			string = ''.join(iter) + ending
			
			# Hardcoded statements seem to be faster than lambdas here.
			if string[3] in '02468' and string[5] in '05':
				if int(string[2:5]) % 3 == 0:
					goodterms.append(int(string))
		            
	return stringify('Terms:', goodterms, '\nSum of all terms:', sum(goodterms))
	
# -----------------------------
# -----------------------------

@SetupAndTime(44,'Smallest pentagon pairs where sums and differences are pentagons:')
def problem44(**kwargs):
		
	def pentagonGen():
		i = 1
		while 1:
			yield i*(3*i-1)/2
			i+=1
	
	existing = set()
	pgen = pentagonGen()
    	
	while 1:
		nextTerm = pgen.next()
		
		for j in existing:
			if nextTerm-j in existing and isPentagon(nextTerm + j):
				return stringify(nextTerm, 'and', j, '- dist:', nextTerm-j)
				
		existing.add(nextTerm)

# -----------------------------
# -----------------------------

@SetupAndTime(45,'Terms that are triangular, hexagonal, and pentagonal:')
def problem45(**kwargs):
	
	def hexagonGen():
		i = 1
		while 1:
			yield i*(2*i-1)
			i+=1
	
	numterms = 3
	terms = []
	words = []
	
	gen = hexagonGen()
	
	counter = 0
	while len(terms) < numterms:
		counter += 1
		next = gen.next()
		if isPentagon(next) and isTrianglegon(next):
		
			terms.append(next)
		
			hexa = str(counter)
			penta = str(int(isPentagon(next, True)))
			tri = str(int(isTrianglegon(next, True)))
			
			maxlen = max(len(hexa), len(penta), len(tri))
		
			z = ' - Term# ' + str(len(terms)) + ': ' + str(next)
			z += '\nHexagon:' + str(hexa)
			z += '\tPentagon:' + str(penta)
			z += '\tTriangle:' + str(tri) + '\n'
			words.append(z)
	
	return EXPAND(words)
			
# -----------------------------
# -----------------------------

@SetupAndTime(46,'The first odd composite number not the sum of a prime and twice a square:')
def problem46(**kwargs):
	
	counter = 13
		
	while 1:
		if not Prime.isPrime(counter):
			valid = False
			
			# Check for +2 (2*(1^2)) as well.
			if counter-2 in Prime.unordered: 
				valid = True
				
			else:			
				for prime in Prime.ordered:
				
					if prime >= counter: break
				
					# Skip prime==2 because otherwise remainder would be odd.
					if prime == 2: continue
					
					# Remainder should be of the form 2*n^2		
					
					remainder = (counter - prime)
					
					divtwo = remainder/2
					
					sqrtdivtwo = int(math.sqrt(divtwo))
					
					if sqrtdivtwo**2 == divtwo and 2*divtwo + prime == counter:
						valid = True
						break
			
			if not valid:
				return counter
		
		counter += 2

# -----------------------------
# -----------------------------

@SetupAndTime(47,'First sequence of %(minNumber) numbers with at least %(minNumber) unique prime factors:', minNumber=4)	
def problem47(**kwargs):

	minNumber = kwargs['minNumber']
	
	found = False
	terms = []
	
	# Sliding window so we don't need to refactor each series of three numbers -
	# in fact, just factor each sequence once.
	i = mul(Prime.ordered[:minNumber])
	adjustedStart = i
		
	while 1:
		
		maxSkip = None
		tmpTerms = []
		
		# Calculate a certain number of new terms based on how much we slid from prior numbers.
		# This [::-1] on the range is a major optimization, as we can determine from the end if
		# the whole series needs to be adjusted, or if only a few positions need to be adjusted,
		# without needing to test every term up to them first.
		
		for j in range(adjustedStart, i+minNumber)[::-1]:
	
			allfactors = set(Prime.primefactors(j))
			if len(allfactors) < minNumber:
				maxSkip = j-i+1
				break
			
			# If this term is invalid, don't try to check the other terms in the series
			tmpTerms.append(allfactors)			
			
		terms = terms + tmpTerms[::-1]
		
		if maxSkip is None:
			seq = '\nSequence: '
			for j in range(minNumber):
				tmpTerm = list(terms[j])
				tmpTerm.sort()
				seq += stringify(str(i+j),  ':', tmpTerm)
				return seq
			
		else:
			terms = terms[maxSkip:]
			i += maxSkip
			adjustedStart = i + len(terms)
			
# -----------------------------
# -----------------------------

@SetupAndTime(48,'Last ten digits in the series n^n for n==1..%(maxTerm):', maxTerm=1000)
def problem48(**kwargs):
	maxTerm = kwargs['maxTerm']
	return str(sum([i**i for i in xrange(maxTerm)]))[-10:]

# -----------------------------
# -----------------------------

@SetupAndTime(49,'4-digit sequence of arithmetic primes:')
def problem49(**kwargs):
	
	numDigits = 4
	
	allNDigitPrimes = []
	allPrimeSet = set()
	
	maxterm = int('1'*numDigits)*9
	toolow = int('1'*(numDigits-1))*9
	
	for prime in Prime.ordered:
		if prime > maxterm: break
		elif prime <= toolow: continue
		else: 
			allNDigitPrimes.append(prime)
			allPrimeSet.add(prime)
			
	found = set()
	for term in allNDigitPrimes:
	
		if term in found: continue
		else:
		
			permutations = [int(''.join(p)) for p in itertools.permutations(str(term))]
			
			primes = list(set([prime for prime in permutations if prime in allPrimeSet]))
			primes.sort()
			
			for baseprime in range(len(primes)-1):
							
				for nextTerm in primes[baseprime+1:]:
					difference = nextTerm - primes[baseprime]
					
					# Expand the sequence
					sequence = [primes[baseprime]]
					valid = True
					while sequence[-1] <= maxterm:
						if not sequence[-1] in primes:
							valid = False
							break
						else:
							sequence.append(sequence[-1] + difference)
							
					if sequence[-1] > maxterm: sequence = sequence[:-1]
					
					if valid and len(sequence) > 2:
						if 1487 in sequence: continue
						return sequence

			for permutation in permutations: found.add(permutation)
			
# -----------------------------
# -----------------------------

@SetupAndTime(50,'largest consecutive prime sum of numbers under %(upto):', upto=1000000)
def problem50(**kwargs):
	upto = kwargs['upto']
	
	# Just keep adding all terms until we surpass the target number.
	# That's the absolute most that could exist.  Then scale up the bottom
	# until we find a target closest to the target, and add to the top as
	# necessary ...
	
	sums = []
	primes = Prime.ordered
	
	for prime in Prime.ordered:
		if len(sums) > 0:
			sums.append(prime + sums[-1])
		else:
			sums.append(prime)
				
		if sums[-1] > upto:
			sums = sums[:-1]
			break
			
	iterations = 0
	found = dict()
	# Go through each sum to see if it's prime, and if so, the number of terms it would need.
	while sum(sums) > 0:
	
		for term in sums:
		
			if term == 0: continue
			
			if Prime.isPrime(term):
				startNum = Prime.ordered[iterations]
				endNum = Prime.ordered[sums.index(term)-iterations] 
				found[term] = [sums.index(term)+1-iterations, startNum, endNum]
				
		sums = [0 if Prime.isPrime(n) else n-Prime.ordered[iterations] for n in sums]
		iterations += 1
		
	largest = (0, 0)
	for elem in found:
		if found[elem][0] > largest[0]:
			largest = (found[elem][0], elem)
			
	result = found[largest[1]]
	return stringify(largest[1], 'starting prime=', result[1], 'ending prime=', result[2], 'num primes in sum=', result[0])
	
# -----------------------------
# -----------------------------

@SetupAndTime(51,'The smallest prime with a series of digits, that when replaced, can create %(seriessize) prime values.', seriessize=8)
def problem51(**kwargs):

	def generateWildcardIndices(digits):
		
		# We don't want a full wildcard string, so omit the final term
		
		for j in xrange(1, int('0b' + '1'*digits, 2)):
		
			term = bin(j)[2:].zfill(digits)
			indices = [position for position in range(len(term)) if term[position] == '1']
			
			term = term.replace('0', 'x')
			yield term, indices
	
	seriesSize = kwargs['seriessize']

	maxDigits = len(str(Prime.ordered[-1]))
		
	# Get lists of n digit primes at a time.
	# Only go up to the maximum value in the prime cache.
	for digits in xrange(2, maxDigits+1):
	
		# Gather all primes in this range.
		minValForNDigits = (int('1'*(digits-1))*9)-1
		maxValForNDigits = int('1'*digits)*9
		
		groupings = []
		
		for prime in Prime.ordered:
		
			if prime < minValForNDigits:   continue
			elif prime > maxValForNDigits: break
			else:                          groupings.append(str(prime))
				
		for test, indices in generateWildcardIndices(digits):
		
			print ' -', test
			
			# Keep track of all numbers matching the patterns we've already searched.
			patterns = set()
			
			tcount = test.count('1')
			
			# Go through each prime and apply the mask to the number.
			for prime in groupings:			

				# If this prime has been tested in earlier permutations, don't test it again.
				if prime in patterns: continue
				
				# If this prime doesn't fit the mask that is provided, don't test it.
				replacement_str = [prime[char] for char in indices]
				if replacement_str.count(replacement_str[0]) != len(replacement_str): continue
							
				# If replacement string ends in a wild card, we can skip even numbers.
				# Additionally, we can skip the term 5 if the length of the string is > 1.
				compareTerm = '1379' if test[-1] == '1' else '0123456789'				
				
				options = []
				for j in compareTerm:
					replacement_str = ''.join([prime[i] if not i in indices else j for i in range(len(prime))])
					if Prime.isPrime(int(replacement_str)):
						if len(str(int(replacement_str))) == len(prime):
							options.append(replacement_str)				
										
				# Ensure that any wildcard matches didn't change the number of digits...
				if len(options) >= seriesSize:
					print '\nSmallest', seriesSize, 'prime value family:', prime
					print 'Replacement mask:', test
					print 'Series:', options
					return

				for match in options: patterns.add(match)
		
# -----------------------------
# -----------------------------
@SetupAndTime(52,'What number multiplied by [1..%(numMultiples)] contains the same digits in each product?', numMultiples=6)
def problem52(**kwargs):

	def infRange(start):
		i = start
		while 1:
			yield i
			i+= 1
			
	def compareDigits(baseTerm, againstTerm):
		
		for c in againstTerm:
			if not c in baseTerm: return False
			
		for c in baseTerm:
			if not c in againstTerm: return False
			
		chardict = dict()
		for char in baseTerm:
			if not char in chardict:
				chardict[char] = 1
			else:
				chardict[char] += 1
				
		for char in againstTerm:
			chardict[char] -= 1
			if chardict[char] < 0:
				return False
				
		return True
	
	# Doesn't work for n > 9 because you'd have 10 digits, and only 9 numbers.
	numMultiples = kwargs['numMultiples']
	
	for i in infRange(1):
	
		istr = str(i)
		nthterms = [str(i*val) for val in xrange(1, numMultiples+1)]
				
		# If n*i is longer in length than this string, then it can't have the exact same digits.
		# As n grows larger, this makes it easier to filter out terms.
		if len(nthterms[-1]) != len(istr): continue
		
		valid = True
		
		for term in nthterms[1:]:
			if not compareDigits(nthterms[0], term):
				valid = False
				break

		if valid:
			return stringify(i, '*', range(1, numMultiples+1), nthterms)

	print 'No result was found.'
	
# -----------------------------
# -----------------------------

@SetupAndTime(53,'Number of combinactions of nCr for 1 <= n <= %(maxn)', maxn = 100)
def problem53(**kwargs):

	# http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
	def choose(n, k):
		"""
		A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
		"""
		if 0 <= k <= n:
			ntok = 1
			ktok = 1
			for t in xrange(1, min(k, n - k) + 1):
				ntok *= n
				ktok *= t
				n -= 1
			return ntok // ktok
		else:
			return 0
			
	greaterThan = 1000000
	maxn = kwargs['maxn']
			
	totals = []
	for i in xrange(1, maxn+1):
		for j in xrange(0, i+1):
			if choose(i, j) > greaterThan:
				totals.append('i='+str(i)+', j='+str(j))
				
	return len(totals)
	
# -----------------------------
# -----------------------------

@SetupAndTime(54,'How many times does player 1 win poker hands?!?!')
def problem54(**kwargs):

	pokerstrings = ["8C TS KC 9H 4S 7D 2S 5D 3S AC", "5C AD 5D AC 9C 7C 5H 8D TD KS", "3H 7H 6S KC JS QH TD JC 2D 8S",
					"TH 8H 5C QS TC 9H 4D JC KS JS", "7C 5H KC QH JD AS KH 4C AD 4S", "5H KS 9C 7D 9H 8D 3S 5D 5C AH",
					"6H 4H 5C 3H 2H 3S QH 5S 6S AS", "TD 8C 4H 7C TC KC 4C 3H 7S KS", "7C 9C 6D KD 3H 4C QS QC AC KH",
					"JC 6S 5H 2H 2D KD 9D 7C AS JS", "AD QH TH 9D 8H TS 6D 3S AS AC", "2H 4S 5C 5S TC KC JD 6C TS 3C",
					"QD AS 6H JS 2C 3D 9H KC 4H 8S", "KD 8S 9S 7C 2S 3S 6D 6S 4H KC", "3C 8C 2D 7D 4D 9S 4S QH 4H JD",
					"8C KC 7S TC 2D TS 8H QD AC 5C", "3D KH QD 6C 6S AD AS 8H 2H QS", "6S 8D 4C 8S 6C QH TC 6D 7D 9D",
					"2S 8D 8C 4C TS 9S 9D 9C AC 3D", "3C QS 2S 4H JH 3D 2D TD 8S 9H", "5H QS 8S 6D 3C 8C JD AS 7H 7D",
					"6H TD 9D AS JH 6C QC 9S KD JC", "AH 8S QS 4D TH AC TS 3C 3D 5C", "5S 4D JS 3D 8H 6C TS 3S AD 8C",
					"6D 7C 5D 5H 3S 5C JC 2H 5S 3D", "5H 6H 2S KS 3D 5D JD 7H JS 8H", "KH 4H AS JS QS QC TC 6D 7C KS",
					"3D QS TS 2H JS 4D AS 9S JC KD", "QD 5H 4D 5D KH 7H 3D JS KD 4H", "2C 9H 6H 5C 9D 6C JC 2D TH 9S",
					"7D 6D AS QD JH 4D JS 7C QS 5C", "3H KH QD AD 8C 8H 3S TH 9D 5S", "AH 9S 4D 9D 8S 4H JS 3C TC 8D",
					"2C KS 5H QD 3S TS 9H AH AD 8S", "5C 7H 5D KD 9H 4D 3D 2D KS AD", "KS KC 9S 6D 2C QH 9D 9H TS TC",
					"9C 6H 5D QH 4D AD 6D QC JS KH", "9S 3H 9D JD 5C 4D 9H AS TC QH", "2C 6D JC 9C 3C AD 9S KH 9D 7D",
					"KC 9C 7C JC JS KD 3H AS 3C 7D", "QD KH QS 2C 3S 8S 8H 9H 9C JC", "QH 8D 3C KC 4C 4H 6D AD 9H 9D",
					"3S KS QS 7H KH 7D 5H 5D JD AD", "2H 2C 6H TH TC 7D 8D 4H 8C AS", "4S 2H AC QC 3S 6D TH 4D 4C KH",
					"4D TC KS AS 7C 3C 6D 2D 9H 6C", "8C TD 5D QS 2C 7H 4C 9C 3H 9H", "5H JH TS 7S TD 6H AD QD 8H 8S",
					"5S AD 9C 8C 7C 8D 5H 9D 8S 2S", "4H KH KS 9S 2S KC 5S AD 4S 7D", "QS 9C QD 6H JS 5D AC 8D 2S AS",
					"KH AC JC 3S 9D 9S 3C 9C 5S JS", "AD 3C 3D KS 3S 5C 9C 8C TS 4S", "JH 8D 5D 6H KD QS QD 3D 6C KC",
					"8S JD 6C 3S 8C TC QC 3C QH JS", "KC JC 8H 2S 9H 9C JH 8S 8C 9S", "8S 2H QH 4D QC 9D KC AS TH 3C",
					"8S 6H TH 7C 2H 6S 3C 3H AS 7S", "QH 5S JS 4H 5H TS 8H AH AC JC", "9D 8H 2S 4S TC JC 3C 7H 3H 5C",
					"3D AD 3C 3S 4C QC AS 5D TH 8C", "6S 9D 4C JS KH AH TS JD 8H AD", "4C 6S 9D 7S AC 4D 3D 3S TC JD",
					"AD 7H 6H 4H JH KC TD TS 7D 6S", "8H JH TC 3S 8D 8C 9S 2C 5C 4D", "2C 9D KC QH TH QS JC 9C 4H TS",
					"QS 3C QD 8H KH 4H 8D TD 8S AC", "7C 3C TH 5S 8H 8C 9C JD TC KD", "QC TC JD TS 8C 3H 6H KD 7C TD",
					"JH QS KS 9C 6D 6S AS 9H KH 6H", "2H 4D AH 2D JH 6H TD 5D 4H JD", "KD 8C 9S JH QD JS 2C QS 5C 7C",
					"4S TC 7H 8D 2S 6H 7S 9C 7C KC", "8C 5D 7H 4S TD QC 8S JS 4H KS", "AD 8S JH 6D TD KD 7C 6C 2D 7D",
					"JC 6H 6S JS 4H QH 9H AH 4C 3C", "6H 5H AS 7C 7S 3D KH KC 5D 5C", "JC 3D TD AS 4D 6D 6S QH JD KS",
					"8C 7S 8S QH 2S JD 5C 7H AH QD", "8S 3C 6H 6C 2C 8D TD 7D 4C 4D", "5D QH KH 7C 2S 7H JS 6D QC QD",
					"AD 6C 6S 7D TH 6H 2H 8H KH 4H", "KS JS KD 5D 2D KH 7D 9C 8C 3D", "9C 6D QD 3C KS 3S 7S AH JD 2D",
					"AH QH AS JC 8S 8H 4C KC TH 7D", "JC 5H TD 7C 5D KD 4C AD 8H JS", "KC 2H AC AH 7D JH KH 5D 7S 6D",
					"9S 5S 9C 6H 8S TD JD 9H 6C AC", "7D 8S 6D TS KD 7H AC 5S 7C 5D", "AH QC JC 4C TC 8C 2H TS 2C 7D",
					"KD KC 6S 3D 7D 2S 8S 3H 5S 5C", "8S 5D 8H 4C 6H KC 3H 7C 5S KD", "JH 8C 3D 3C 6C KC TD 7H 7C 4C",
					"JC KC 6H TS QS TD KS 8H 8C 9S", "6C 5S 9C QH 7D AH KS KC 9S 2C", "4D 4S 8H TD 9C 3S 7D 9D AS TH",
					"6S 7D 3C 6H 5D KD 2C 5C 9D 9C", "2H KC 3D AD 3H QD QS 8D JC 4S", "8C 3H 9C 7C AD 5D JC 9D JS AS",
					"5D 9H 5C 7H 6S 6C QC JC QD 9S", "JC QS JH 2C 6S 9C QC 3D 4S TC", "4H 5S 8D 3D 4D 2S KC 2H JS 2C",
					"TD 3S TH KD 4D 7H JH JS KS AC", "7S 8C 9S 2D 8S 7D 5C AD 9D AS", "8C 7H 2S 6C TH 3H 4C 3S 8H AC",
					"KD 5H JC 8H JD 2D 4H TD JH 5C", "3D AS QH KS 7H JD 8S 5S 6D 5H", "9S 6S TC QS JC 5C 5D 9C TH 8C",
					"5H 3S JH 9H 2S 2C 6S 7S AS KS", "8C QD JC QS TC QC 4H AC KH 6C", "TC 5H 7D JH 4H 2H 8D JC KS 4D",
					"5S 9C KH KD 9H 5C TS 3D 7D 2D", "5H AS TC 4D 8C 2C TS 9D 3H 8D", "6H 8D 2D 9H JD 6C 4S 5H 5S 6D",
					"AD 9C JC 7D 6H 9S 6D JS 9H 3C", "AD JH TC QS 4C 5D 9S 7C 9C AH", "KD 6H 2H TH 8S QD KS 9D 9H AS",
					"4H 8H 8D 5H 6C AH 5S AS AD 8S", "QS 5D 4S 2H TD KS 5H AC 3H JC", "9C 7D QD KD AC 6D 5H QH 6H 5S",
					"KC AH QH 2H 7D QS 3H KS 7S JD", "6C 8S 3H 6D KS QD 5D 5C 8H TC", "9H 4D 4S 6S 9D KH QC 4H 6C JD",
					"TD 2D QH 4S 6H JH KD 3C QD 8C", "4S 6H 7C QD 9D AS AH 6S AD 3C", "2C KC TH 6H 8D AH 5C 6D 8S 5D",
					"TD TS 7C AD JC QD 9H 3C KC 7H", "5D 4D 5S 8H 4H 7D 3H JD KD 2D", "JH TD 6H QS 4S KD 5C 8S 7D 8H",
					"AC 3D AS 8C TD 7H KH 5D 6C JD", "9D KS 7C 6D QH TC JD KD AS KC", "JH 8S 5S 7S 7D AS 2D 3D AD 2H",
					"2H 5D AS 3C QD KC 6H 9H 9S 2C", "9D 5D TH 4C JH 3H 8D TC 8H 9H", "6H KD 2C TD 2H 6C 9D 2D JS 8C",
					"KD 7S 3C 7C AS QH TS AD 8C 2S", "QS 8H 6C JS 4C 9S QC AD TD TS", "2H 7C TS TC 8C 3C 9H 2D 6D JC",
					"TC 2H 8D JH KS 6D 3H TD TH 8H", "9D TD 9H QC 5D 6C 8H 8C KC TS", "2H 8C 3D AH 4D TH TC 7D 8H KC",
					"TS 5C 2D 8C 6S KH AH 5H 6H KC", "5S 5D AH TC 4C JD 8D 6H 8C 6C", "KC QD 3D 8H 2D JC 9H 4H AD 2S",
					"TD 6S 7D JS KD 4H QS 2S 3S 8C", "4C 9H JH TS 3S 4H QC 5S 9S 9C", "2C KD 9H JS 9S 3H JC TS 5D AC",
					"AS 2H 5D AD 5H JC 7S TD JS 4C", "2D 4S 8H 3D 7D 2C AD KD 9C TS", "7H QD JH 5H JS AC 3D TH 4C 8H",
					"6D KH KC QD 5C AD 7C 2D 4H AC", "3D 9D TC 8S QD 2C JC 4H JD AH", "6C TD 5S TC 8S AH 2C 5D AS AC",
					"TH 7S 3D AS 6C 4C 7H 7D 4H AH", "5C 2H KS 6H 7S 4H 5H 3D 3C 7H", "3C 9S AC 7S QH 2H 3D 6S 3S 3H",
					"2D 3H AS 2C 6H TC JS 6S 9C 6C", "QH KD QD 6D AC 6H KH 2C TS 8C", "8H 7D 3S 9H 5D 3H 4S QC 9S 5H",
					"2D 9D 7H 6H 3C 8S 5H 4D 3S 4S", "KD 9S 4S TC 7S QC 3S 8S 2H 7H", "TC 3D 8C 3H 6C 2H 6H KS KD 4D",
					"KC 3D 9S 3H JS 4S 8H 2D 6C 8S", "6H QS 6C TC QD 9H 7D 7C 5H 4D", "TD 9D 8D 6S 6C TC 5D TS JS 8H",
					"4H KC JD 9H TC 2C 6S 5H 8H AS", "JS 9C 5C 6S 9D JD 8H KC 4C 6D", "4D 8D 8S 6C 7C 6H 7H 8H 5C KC",
					"TC 3D JC 6D KS 9S 6H 7S 9C 2C", "6C 3S KD 5H TS 7D 9H 9S 6H KH", "3D QD 4C 6H TS AC 3S 5C 2H KD",
					"4C AS JS 9S 7C TS 7H 9H JC KS", "4H 8C JD 3H 6H AD 9S 4S 5S KS", "4C 2C 7D 3D AS 9C 2S QS KC 6C",
					"8S 5H 3D 2S AC 9D 6S 3S 4D TD", "QD TH 7S TS 3D AC 7H 6C 5D QC", "TC QD AD 9C QS 5C 8D KD 3D 3C",
					"9D 8H AS 3S 7C 8S JD 2D 8D KC", "4C TH AC QH JS 8D 7D 7S 9C KH", "9D 8D 4C JH 2C 2S QD KD TS 4H",
					"4D 6D 5D 2D JH 3S 8S 3H TC KH", "AD 4D 2C QS 8C KD JH JD AH 5C", "5C 6C 5H 2H JH 4H KS 7C TC 3H",
					"3C 4C QC 5D JH 9C QD KH 8D TC", "3H 9C JS 7H QH AS 7C 9H 5H JC", "2D 5S QD 4S 3C KC 6S 6C 5C 4C",
					"5D KH 2D TS 8S 9C AS 9S 7C 4C", "7C AH 8C 8D 5S KD QH QS JH 2C", "8C 9D AH 2H AC QC 5S 8H 7H 2C",
					"QD 9H 5S QS QC 9C 5H JC TH 4H", "6C 6S 3H 5H 3S 6H KS 8D AC 7S", "AC QH 7H 8C 4S KC 6C 3D 3S TC",
					"9D 3D JS TH AC 5H 3H 8S 3S TC", "QD KH JS KS 9S QC 8D AH 3C AC", "5H 6C KH 3S 9S JH 2D QD AS 8C",
					"6C 4D 7S 7H 5S JC 6S 9H 4H JH", "AH 5S 6H 9S AD 3S TH 2H 9D 8C", "4C 8D 9H 7C QC AD 4S 9C KC 5S",
					"9D 6H 4D TC 4C JH 2S 5D 3S AS", "2H 6C 7C KH 5C AD QS TH JD 8S", "3S 4S 7S AH AS KC JS 2S AD TH",
					"JS KC 2S 7D 8C 5C 9C TS 5H 9D", "7S 9S 4D TD JH JS KH 6H 5D 2C", "JD JS JC TH 2D 3D QD 8C AC 5H",
					"7S KH 5S 9D 5D TD 4S 6H 3C 2D", "4S 5D AC 8D 4D 7C AD AS AH 9C", "6S TH TS KS 2C QC AH AS 3C 4S",
					"2H 8C 3S JC 5C 7C 3H 3C KH JH", "7S 3H JC 5S 6H 4C 2S 4D KC 7H", "4D 7C 4H 9S 8S 6S AD TC 6C JC",
					"KH QS 3S TC 4C 8H 8S AC 3C TS", "QD QS TH 3C TS 7H 7D AH TD JC", "TD JD QC 4D 9S 7S TS AD 7D AC",
					"AH 7H 4S 6D 7C 2H 9D KS JC TD", "7C AH JD 4H 6D QS TS 2H 2C 5C", "TC KC 8C 9S 4C JS 3C JC 6S AH",
					"AS 7D QC 3D 5S JC JD 9D TD KH", "TH 3C 2S 6H AH AC 5H 5C 7S 8H", "QC 2D AC QD 2S 3S JD QS 6S 8H",
					"KC 4H 3C 9D JS 6H 3S 8S AS 8C", "7H KC 7D JD 2H JC QH 5S 3H QS", "9H TD 3S 8H 7S AC 5C 6C AH 7C",
					"8D 9H AH JD TD QS 7D 3S 9C 8S", "AH QH 3C JD KC 4S 5S 5D TD KS", "9H 7H 6S JH TH 4C 7C AD 5C 2D",
					"7C KD 5S TC 9D 6S 6C 5D 2S TH", "KC 9H 8D 5H 7H 4H QC 3D 7C AS", "6S 8S QC TD 4S 5C TH QS QD 2S",
					"8S 5H TH QC 9H 6S KC 7D 7C 5C", "7H KD AH 4D KH 5C 4S 2D KC QH", "6S 2C TD JC AS 4D 6C 8C 4H 5S",
					"JC TC JD 5S 6S 8D AS 9D AD 3S", "6D 6H 5D 5S TC 3D 7D QS 9D QD", "4S 6C 8S 3S 7S AD KS 2D 7D 7C",
					"KC QH JC AC QD 5D 8D QS 7H 7D", "JS AH 8S 5H 3D TD 3H 4S 6C JH", "4S QS 7D AS 9H JS KS 6D TC 5C",
					"2D 5C 6H TC 4D QH 3D 9H 8S 6C", "6D 7H TC TH 5S JD 5C 9C KS KD", "8D TD QH 6S 4S 6C 8S KC 5C TC",
					"5S 3D KS AC 4S 7D QD 4C TH 2S", "TS 8H 9S 6S 7S QH 3C AH 7H 8C", "4C 8C TS JS QC 3D 7D 5D 7S JH",
					"8S 7S 9D QC AC 7C 6D 2H JH KC", "JS KD 3C 6S 4S 7C AH QC KS 5H", "KS 6S 4H JD QS TC 8H KC 6H AS",
					"KH 7C TC 6S TD JC 5C 7D AH 3S", "3H 4C 4H TC TH 6S 7H 6D 9C QH", "7D 5H 4S 8C JS 4D 3D 8S QH KC",
					"3H 6S AD 7H 3S QC 8S 4S 7S JS", "3S JD KH TH 6H QS 9C 6C 2D QD", "4S QH 4D 5H KC 7D 6D 8D TH 5S",
					"TD AD 6S 7H KD KH 9H 5S KC JC", "3H QC AS TS 4S QD KS 9C 7S KC", "TS 6S QC 6C TH TC 9D 5C 5D KD",
					"JS 3S 4H KD 4C QD 6D 9S JC 9D", "8S JS 6D 4H JH 6H 6S 6C KS KH", "AC 7D 5D TC 9S KH 6S QD 6H AS",
					"AS 7H 6D QH 8D TH 2S KH 5C 5H", "4C 7C 3D QC TC 4S KH 8C 2D JS", "6H 5D 7S 5H 9C 9H JH 8S TH 7H",
					"AS JS 2S QD KH 8H 4S AC 8D 8S", "3H 4C TD KD 8C JC 5C QS 2D JD", "TS 7D 5D 6C 2C QS 2H 3C AH KS",
					"4S 7C 9C 7D JH 6C 5C 8H 9D QD", "2S TD 7S 6D 9C 9S QS KH QH 5C", "JC 6S 9C QH JH 8D 7S JS KH 2H",
					"8D 5H TH KC 4D 4S 3S 6S 3D QS", "2D JD 4C TD 7C 6D TH 7S JC AH", "QS 7S 4C TH 9D TS AD 4D 3H 6H",
					"2D 3H 7D JD 3D AS 2S 9C QC 8S", "4H 9H 9C 2C 7S JH KD 5C 5D 6H", "TC 9H 8H JC 3C 9S 8D KS AD KC",
					"TS 5H JD QS QH QC 8D 5D KH AH", "5D AS 8S 6S 4C AH QC QD TH 7H", "3H 4H 7D 6S 4S 9H AS 8H JS 9D",
					"JD 8C 2C 9D 7D 5H 5S 9S JC KD", "KD 9C 4S QD AH 7C AD 9D AC TD", "6S 4H 4S 9C 8D KS TC 9D JH 7C",
					"5S JC 5H 4S QH AC 2C JS 2S 9S", "8C 5H AS QD AD 5C 7D 8S QC TD", "JC 4C 8D 5C KH QS 4D 6H 2H 2C",
					"TH 4S 2D KC 3H QD AC 7H AD 9D", "KH QD AS 8H TH KC 8D 7S QH 8C", "JC 6C 7D 8C KH AD QS 2H 6S 2D",
					"JC KH 2D 7D JS QC 5H 4C 5D AD", "TS 3S AD 4S TD 2D TH 6S 9H JH", "9H 2D QS 2C 4S 3D KH AS AC 9D",
					"KH 6S 8H 4S KD 7D 9D TS QD QC", "JH 5H AH KS AS AD JC QC 5S KH", "5D 7D 6D KS KD 3D 7C 4D JD 3S",
					"AC JS 8D 5H 9C 3H 4H 4D TS 2C", "6H KS KH 9D 7C 2S 6S 8S 2H 3D", "6H AC JS 7S 3S TD 8H 3H 4H TH",
					"9H TC QC KC 5C KS 6H 4H AC 8S", "TC 7D QH 4S JC TS 6D 6C AC KH", "QH 7D 7C JH QS QD TH 3H 5D KS",
					"3D 5S 8D JS 4C 2C KS 7H 9C 4H", "5H 8S 4H TD 2C 3S QD QC 3H KC", "QC JS KD 9C AD 5S 9D 7D 7H TS",
					"8C JC KH 7C 7S 6C TS 2C QD TH", "5S 9D TH 3C 7S QH 8S 9C 2H 5H", "5D 9H 6H 2S JS KH 3H 7C 2H 5S",
					"JD 5D 5S 2C TC 2S 6S 6C 3C 8S", "4D KH 8H 4H 2D KS 3H 5C 2S 9H", "3S 2D TD 7H 8S 6H JD KC 9C 8D",
					"6S QD JH 7C 9H 5H 8S 8H TH TD", "QS 7S TD 7D TS JC KD 7C 3C 2C", "3C JD 8S 4H 2D 2S TD AS 4D AC",
					"AH KS 6C 4C 4S 7D 8C 9H 6H AS", "5S 3C 9S 2C QS KD 4D 4S AC 5D", "2D TS 2C JS KH QH 5D 8C AS KC",
					"KD 3H 6C TH 8S 7S KH 6H 9S AC", "6H 7S 6C QS AH 2S 2H 4H 5D 5H", "5H JC QD 2C 2S JD AS QC 6S 7D",
					"6C TC AS KD 8H 9D 2C 7D JH 9S", "2H 4C 6C AH 8S TD 3H TH 7C TS", "KD 4S TS 6C QH 8D 9D 9C AH 7D",
					"6D JS 5C QD QC 9C 5D 8C 2H KD", "3C QH JH AD 6S AH KC 8S 6D 6H", "3D 7C 4C 7S 5S 3S 6S 5H JC 3C",
					"QH 7C 5H 3C 3S 8C TS 4C KD 9C", "QD 3S 7S 5H 7H QH JC 7C 8C KD", "3C KD KH 2S 4C TS AC 6S 2C 7C",
					"2C KH 3C 4C 6H 4D 5H 5S 7S QD", "4D 7C 8S QD TS 9D KS 6H KD 3C", "QS 4D TS 7S 4C 3H QD 8D 9S TC",
					"TS QH AC 6S 3C 9H 9D QS 8S 6H", "3S 7S 5D 4S JS 2D 6C QH 6S TH", "4C 4H AS JS 5D 3D TS 9C AC 8S",
					"6S 9C 7C 3S 5C QS AD AS 6H 3C", "9S 8C 7H 3H 6S 7C AS 9H JD KH", "3D 3H 7S 4D 6C 7C AC 2H 9C TH",
					"4H 5S 3H AC TC TH 9C 9H 9S 8D", "8D 9H 5H 4D 6C 2H QD 6S 5D 3S", "4C 5C JD QS 4D 3H TH AC QH 8C",
					"QC 5S 3C 7H AD 4C KS 4H JD 6D", "QS AH 3H KS 9H 2S JS JH 5H 2H", "2H 5S TH 6S TS 3S KS 3C 5H JS",
					"2D 9S 7H 3D KC JH 6D 7D JS TD", "AC JS 8H 2C 8C JH JC 2D TH 7S", "5D 9S 8H 2H 3D TC AH JC KD 9C",
					"9D QD JC 2H 6D KH TS 9S QH TH", "2C 8D 4S JD 5H 3H TH TC 9C KC", "AS 3D 9H 7D 4D TH KH 2H 7S 3H",
					"4H 7S KS 2S JS TS 8S 2H QD 8D", "5S 6H JH KS 8H 2S QC AC 6S 3S", "JC AS AD QS 8H 6C KH 4C 4D QD",
					"2S 3D TS TD 9S KS 6S QS 5C 8D", "3C 6D 4S QC KC JH QD TH KH AD", "9H AH 4D KS 2S 8D JH JC 7C QS",
					"2D 6C TH 3C 8H QD QH 2S 3S KS", "6H 5D 9S 4C TS TD JS QD 9D JD", "5H 8H KH 8S KS 7C TD AD 4S KD",
					"2C 7C JC 5S AS 6C 7D 8S 5H 9C", "6S QD 9S TS KH QS 5S QH 3C KC", "7D 3H 3C KD 5C AS JH 7H 6H JD",
					"9D 5C 9H KC 8H KS 4S AD 4D 2S", "3S JD QD 8D 2S 7C 5S 6S 5H TS", "6D 9S KC TD 3S 6H QD JD 5C 8D",
					"5H 9D TS KD 8D 6H TD QC 4C 7D", "6D 4S JD 9D AH 9S AS TD 9H QD", "2D 5S 2H 9C 6H 9S TD QC 7D TC",
					"3S 2H KS TS 2C 9C 8S JS 9D 7D", "3C KC 6D 5D 6C 6H 8S AS 7S QS", "JH 9S 2H 8D 4C 8H 9H AD TH KH",
					"QC AS 2S JS 5C 6H KD 3H 7H 2C", "QD 8H 2S 8D 3S 6D AH 2C TC 5C", "JD JS TS 8S 3H 5D TD KC JC 6H",
					"6S QS TC 3H 5D AH JC 7C 7D 4H", "7C 5D 8H 9C 2H 9H JH KH 5S 2C", "9C 7H 6S TH 3S QC QD 4C AC JD",
					"2H 5D 9S 7D KC 3S QS 2D AS KH", "2S 4S 2H 7D 5C TD TH QH 9S 4D", "6D 3S TS 6H 4H KS 9D 8H 5S 2D",
					"9H KS 4H 3S 5C 5D KH 6H 6S JS", "KC AS 8C 4C JC KH QC TH QD AH", "6S KH 9S 2C 5H TC 3C 7H JC 4D",
					"JD 4S 6S 5S 8D 7H 7S 4D 4C 2H", "7H 9H 5D KH 9C 7C TS TC 7S 5H", "4C 8D QC TS 4S 9H 3D AD JS 7C",
					"8C QS 5C 5D 3H JS AH KC 4S 9D", "TS JD 8S QS TH JH KH 2D QD JS", "JD QC 5D 6S 9H 3S 2C 8H 9S TS",
					"2S 4C AD 7H JC 5C 2D 6D 4H 3D", "7S JS 2C 4H 8C AD QD 9C 3S TD", "JD TS 4C 6H 9H 7D QD 6D 3C AS",
					"AS 7C 4C 6S 5D 5S 5C JS QC 4S", "KD 6S 9S 7C 3C 5S 7D JH QD JS", "4S 7S JH 2C 8S 5D 7H 3D QH AD",
					"TD 6H 2H 8D 4H 2D 7C AD KH 5D", "TS 3S 5H 2C QD AH 2S 5C KH TD", "KC 4D 8C 5D AS 6C 2H 2S 9H 7C",
					"KD JS QC TS QS KH JH 2C 5D AD", "3S 5H KC 6C 9H 3H 2H AD 7D 7S", "7S JS JH KD 8S 7D 2S 9H 7C 2H",
					"9H 2D 8D QC 6S AD AS 8H 5H 6C", "2S 7H 6C 6D 7D 8C 5D 9D JC 3C", "7C 9C 7H JD 2H KD 3S KH AD 4S",
					"QH AS 9H 4D JD KS KD TS KH 5H", "4C 8H 5S 3S 3D 7D TD AD 7S KC", "JS 8S 5S JC 8H TH 9C 4D 5D KC",
					"7C 5S 9C QD 2C QH JS 5H 8D KH", "TD 2S KS 3D AD KC 7S TC 3C 5D", "4C 2S AD QS 6C 9S QD TH QH 5C",
					"8C AD QS 2D 2S KC JD KS 6C JC", "8D 4D JS 2H 5D QD 7S 7D QH TS", "6S 7H 3S 8C 8S 9D QS 8H 6C 9S",
					"4S TC 2S 5C QD 4D QS 6D TH 6S", "3S 5C 9D 6H 8D 4C 7D TC 7C TD", "AH 6S AS 7H 5S KD 3H 5H AC 4C",
					"8D 8S AH KS QS 2C AD 6H 7D 5D", "6H 9H 9S 2H QS 8S 9C 5D 2D KD", "TS QC 5S JH 7D 7S TH 9S 9H AC",
					"7H 3H 6S KC 4D 6D 5C 4S QD TS", "TD 2S 7C QD 3H JH 9D 4H 7S 7H", "KS 3D 4H 5H TC 2S AS 2D 6D 7D",
					"8H 3C 7H TD 3H AD KC TH 9C KH", "TC 4C 2C 9S 9D 9C 5C 2H JD 3C", "3H AC TS 5D AD 8D 6H QC 6S 8C",
					"2S TS 3S JD 7H 8S QH 4C 5S 8D", "AC 4S 6C 3C KH 3D 7C 2D 8S 2H", "4H 6C 8S TH 2H 4S 8H 9S 3H 7S",
					"7C 4C 9C 2C 5C AS 5D KD 4D QH", "9H 4H TS AS 7D 8D 5D 9S 8C 2H", "QC KD AC AD 2H 7S AS 3S 2D 9S",
					"2H QC 8H TC 6D QD QS 5D KH 3C", "TH JD QS 4C 2S 5S AD 7H 3S AS", "7H JS 3D 6C 3S 6D AS 9S AC QS",
					"9C TS AS 8C TC 8S 6H 9D 8D 6C", "4D JD 9C KC 7C 6D KS 3S 8C AS", "3H 6S TC 8D TS 3S KC 9S 7C AS",
					"8C QC 4H 4S 8S 6C 3S TC AH AC", "4D 7D 5C AS 2H 6S TS QC AD TC", "QD QC 8S 4S TH 3D AH TS JH 4H",
					"5C 2D 9S 2C 3H 3C 9D QD QH 7D", "KC 9H 6C KD 7S 3C 4D AS TC 2D", "3D JS 4D 9D KS 7D TH QC 3H 3C",
					"8D 5S 2H 9D 3H 8C 4C 4H 3C TH", "JC TH 4S 6S JD 2D 4D 6C 3D 4C", "TS 3S 2D 4H AC 2C 6S 2H JH 6H",
					"TD 8S AD TC AH AC JH 9S 6S 7S", "6C KC 4S JD 8D 9H 5S 7H QH AH", "KD 8D TS JH 5C 5H 3H AD AS JS",
					"2D 4H 3D 6C 8C 7S AD 5D 5C 8S", "TD 5D 7S 9C 4S 5H 6C 8C 4C 8S", "JS QH 9C AS 5C QS JC 3D QC 7C",
					"JC 9C KH JH QS QC 2C TS 3D AD", "5D JH AC 5C 9S TS 4C JD 8C KS", "KC AS 2D KH 9H 2C 5S 4D 3D 6H",
					"TH AH 2D 8S JC 3D 8C QH 7S 3S", "8H QD 4H JC AS KH KS 3C 9S 6D", "9S QH 7D 9C 4S AC 7H KH 4D KD",
					"AH AD TH 6D 9C 9S KD KS QH 4H", "QD 6H 9C 7C QS 6D 6S 9D 5S JH", "AH 8D 5H QD 2H JC KS 4H KH 5S",
					"5C 2S JS 8D 9C 8C 3D AS KC AH", "JD 9S 2H QS 8H 5S 8C TH 5C 4C", "QC QS 8C 2S 2C 3S 9C 4C KS KH",
					"2D 5D 8S AH AD TD 2C JS KS 8C", "TC 5S 5H 8H QC 9H 6H JD 4H 9S", "3C JH 4H 9H AH 4S 2H 4C 8D AC",
					"8S TH 4D 7D 6D QD QS 7S TC 7C", "KH 6D 2D JD 5H JS QD JH 4H 4S", "9C 7S JH 4S 3S TS QC 8C TC 4H",
					"QH 9D 4D JH QS 3S 2C 7C 6C 2D", "4H 9S JD 5C 5H AH 9D TS 2D 4C", "KS JH TS 5D 2D AH JS 7H AS 8D",
					"JS AH 8C AD KS 5S 8H 2C 6C TH", "2H 5D AD AC KS 3D 8H TS 6H QC", "6D 4H TS 9C 5H JS JH 6S JD 4C",
					"JH QH 4H 2C 6D 3C 5D 4C QS KC", "6H 4H 6C 7H 6S 2S 8S KH QC 8C", "3H 3D 5D KS 4H TD AD 3S 4D TS",
					"5S 7C 8S 7D 2C KS 7S 6C 8C JS", "5D 2H 3S 7C 5C QD 5H 6D 9C 9H", "JS 2S KD 9S 8D TD TS AC 8C 9D",
					"5H QD 2S AC 8C 9H KS 7C 4S 3C", "KH AS 3H 8S 9C JS QS 4S AD 4D", "AS 2S TD AD 4D 9H JC 4C 5H QS",
					"5D 7C 4H TC 2D 6C JS 4S KC 3S", "4C 2C 5D AC 9H 3D JD 8S QS QH", "2C 8S 6H 3C QH 6D TC KD AC AH",
					"QC 6C 3S QS 4S AC 8D 5C AD KH", "5S 4C AC KH AS QC 2C 5C 8D 9C", "8H JD 3C KH 8D 5C 9C QD QH 9D",
					"7H TS 2C 8C 4S TD JC 9C 5H QH", "JS 4S 2C 7C TH 6C AS KS 7S JD", "JH 7C 9H 7H TC 5H 3D 6D 5D 4D",
					"2C QD JH 2H 9D 5S 3D TD AD KS", "JD QH 3S 4D TH 7D 6S QS KS 4H", "TC KS 5S 8D 8H AD 2S 2D 4C JH",
					"5S JH TC 3S 2D QS 9D 4C KD 9S", "AC KH 3H AS 9D KC 9H QD 6C 6S", "9H 7S 3D 5C 7D KC TD 8H 4H 6S",
					"3C 7H 8H TC QD 4D 7S 6S QH 6C", "6D AD 4C QD 6C 5D 7D 9D KS TS", "JH 2H JD 9S 7S TS KH 8D 5D 8H",
					"2D 9S 4C 7D 9D 5H QD 6D AC 6S", "7S 6D JC QD JH 4C 6S QS 2H 7D", "8C TD JH KD 2H 5C QS 2C JS 7S",
					"TC 5H 4H JH QD 3S 5S 5D 8S KH", "KS KH 7C 2C 5D JH 6S 9C 6D JC", "5H AH JD 9C JS KC 2H 6H 4D 5S",
					"AS 3C TH QC 6H 9C 8S 8C TD 7C", "KC 2C QD 9C KH 4D 7S 3C TS 9H", "9C QC 2S TS 8C TD 9S QD 3S 3C",
					"4D 9D TH JH AH 6S 2S JD QH JS", "QD 9H 6C KD 7D 7H 5D 6S 8H AH", "8H 3C 4S 2H 5H QS QH 7S 4H AC",
					"QS 3C 7S 9S 4H 3S AH KS 9D 7C", "AD 5S 6S 2H 2D 5H TC 4S 3C 8C", "QH TS 6S 4D JS KS JH AS 8S 6D",
					"2C 8S 2S TD 5H AS TC TS 6C KC", "KC TS 8H 2H 3H 7C 4C 5S TH TD", "KD AD KH 7H 7S 5D 5H 5S 2D 9C",
					"AD 9S 3D 7S 8C QC 7C 9C KD KS", "3C QC 9S 8C 4D 5C AS QD 6C 2C", "2H KC 8S JD 7S AC 8D 5C 2S 4D",
					"9D QH 3D 2S TC 3S KS 3C 9H TD", "KD 6S AC 2C 7H 5H 3S 6C 6H 8C", "QH TC 8S 6S KH TH 4H 5D TS 4D",
					"8C JS 4H 6H 2C 2H 7D AC QD 3D", "QS KC 6S 2D 5S 4H TD 3H JH 4C", "7S 5H 7H 8H KH 6H QS TH KD 7D",
					"5H AD KD 7C KH 5S TD 6D 3C 6C", "8C 9C 5H JD 7C KC KH 7H 2H 3S", "7S 4H AD 4D 8S QS TH 3D 7H 5S",
					"8D TC KS KD 9S 6D AD JD 5C 2S", "7H 8H 6C QD 2H 6H 9D TC 9S 7C", "8D 6D 4C 7C 6C 3C TH KH JS JH",
					"5S 3S 8S JS 9H AS AD 8H 7S KD", "JH 7C 2C KC 5H AS AD 9C 9S JS", "AD AC 2C 6S QD 7C 3H TH KS KD",
					"9D JD 4H 8H 4C KH 7S TS 8C KC", "3S 5S 2H 7S 6H 7D KS 5C 6D AD", "5S 8C 9H QS 7H 7S 2H 6C 7D TD",
					"QS 5S TD AC 9D KC 3D TC 2D 4D", "TD 2H 7D JD QD 4C 7H 5D KC 3D", "4C 3H 8S KD QH 5S QC 9H TC 5H",
					"9C QD TH 5H TS 5C 9H AH QH 2C", "4D 6S 3C AC 6C 3D 2C 2H TD TH", "AC 9C 5D QC 4D AD 8D 6D 8C KC",
					"AD 3C 4H AC 8D 8H 7S 9S TD JC", "4H 9H QH JS 2D TH TD TC KD KS", "5S 6S 9S 8D TH AS KH 5H 5C 8S",
					"JD 2S 9S 6S 5S 8S 5D 7S 7H 9D", "5D 8C 4C 9D AD TS 2C 7D KD TC", "8S QS 4D KC 5C 8D 4S KH JD KD",
					"AS 5C AD QH 7D 2H 9S 7H 7C TC", "2S 8S JD KH 7S 6C 6D AD 5D QC", "9H 6H 3S 8C 8H AH TC 4H JS TD",
					"2C TS 4D 7H 2D QC 9C 5D TH 7C", "6C 8H QC 5D TS JH 5C 5H 9H 4S", "2D QC 7H AS JS 8S 2H 4C 4H 8D",
					"JS 6S AC KD 3D 3C 4S 7H TH KC", "QH KH 6S QS 5S 4H 3C QD 3S 3H", "7H AS KH 8C 4H 9C 5S 3D 6S TS",
					"9C 7C 3H 5S QD 2C 3D AD AC 5H", "JH TD 2D 4C TS 3H KH AD 3S 7S", "AS 4C 5H 4D 6S KD JC 3C 6H 2D",
					"3H 6S 8C 2D TH 4S AH QH AD 5H", "7C 2S 9H 7H KC 5C 6D 5S 3H JC", "3C TC 9C 4H QD TD JH 6D 9H 5S",
					"7C 6S 5C 5D 6C 4S 7H 9H 6H AH", "AD 2H 7D KC 2C 4C 2S 9S 7H 3S", "TH 4C 8S 6S 3S AD KS AS JH TD",
					"5C TD 4S 4D AD 6S 5D TC 9C 7D", "8H 3S 4D 4S 5S 6H 5C AC 3H 3D", "9H 3C AC 4S QS 8S 9D QH 5H 4D",
					"JC 6C 5H TS AC 9C JD 8C 7C QD", "8S 8H 9C JD 2D QC QH 6H 3C 8D", "KS JS 2H 6H 5H QH QS 3H 7C 6D",
					"TC 3H 4S 7H QC 2H 3S 8C JS KH", "AH 8H 5S 4C 9H JD 3H 7S JC AC", "3C 2D 4C 5S 6C 4S QS 3S JD 3D",
					"5H 2D TC AH KS 6D 7H AD 8C 6H", "6C 7S 3C JD 7C 8H KS KH AH 6D", "AH 7D 3H 8H 8S 7H QS 5H 9D 2D",
					"JD AC 4H 7S 8S 9S KS AS 9D QH", "7S 2C 8S 5S JH QS JC AH KD 4C", "AH 2S 9H 4H 8D TS TD 6H QH JD",
					"4H JC 3H QS 6D 7S 9C 8S 9D 8D", "5H TD 4S 9S 4C 8C 8D 7H 3H 3D", "QS KH 3S 2C 2S 3C 7S TD 4S QD",
					"7C TD 4D 5S KH AC AS 7H 4C 6C", "2S 5H 6D JD 9H QS 8S 2C 2H TD", "2S TS 6H 9H 7S 4H JC 4C 5D 5S",
					"2C 5H 7D 4H 3S QH JC JS 6D 8H", "4C QH 7C QD 3S AD TH 8S 5S TS", "9H TC 2S TD JC 7D 3S 3D TH QH",
					"7D 4C 8S 5C JH 8H 6S 3S KC 3H", "JC 3H KH TC QH TH 6H 2C AC 5H", "QS 2H 9D 2C AS 6S 6C 2S 8C 8S",
					"9H 7D QC TH 4H KD QS AC 7S 3C", "4D JH 6S 5S 8H KS 9S QC 3S AS", "JD 2D 6S 7S TC 9H KC 3H 7D KD",
					"2H KH 7C 4D 4S 3H JS QD 7D KC", "4C JC AS 9D 3C JS 6C 8H QD 4D", "AH JS 3S 6C 4C 3D JH 6D 9C 9H",
					"9H 2D 8C 7H 5S KS 6H 9C 2S TC", "6C 8C AD 7H 6H 3D KH AS 5D TH", "KS 8C 3S TS 8S 4D 5S 9S 6C 4H",
					"9H 4S 4H 5C 7D KC 2D 2H 9D JH", "5C JS TC 9D 9H 5H 7S KH JC 6S", "7C 9H 8H 4D JC KH JD 2H TD TC",
					"8H 6C 2H 2C KH 6H 9D QS QH 5H", "AC 7D 2S 3D QD JC 2D 8D JD JH", "2H JC 2D 7H 2C 3C 8D KD TD 4H",
					"3S 4H 6D 8D TS 3H TD 3D 6H TH", "JH JC 3S AC QH 9H 7H 8S QC 2C", "7H TD QS 4S 8S 9C 2S 5D 4D 2H",
					"3D TS 3H 2S QC 8H 6H KC JC KS", "5D JD 7D TC 8C 6C 9S 3D 8D AC", "8H 6H JH 6C 5D 8D 8S 4H AD 2C",
					"9D 4H 2D 2C 3S TS AS TC 3C 5D", "4D TH 5H KS QS 6C 4S 2H 3D AD", "5C KC 6H 2C 5S 3C 4D 2D 9H 9S",
					"JD 4C 3H TH QH 9H 5S AH 8S AC", "7D 9S 6S 2H TD 9C 4H 8H QS 4C", "3C 6H 5D 4H 8C 9C KC 6S QD QS",
					"3S 9H KD TC 2D JS 8C 6S 4H 4S", "2S 4C 8S QS 6H KH 3H TH 8C 5D", "2C KH 5S 3S 7S 7H 6C 9D QD 8D",
					"8H KS AC 2D KH TS 6C JS KC 7H", "9C KS 5C TD QC AH 6C 5H 9S 7C", "5D 4D 3H 4H 6S 7C 7S AH QD TD",
					"2H 7D QC 6S TC TS AH 7S 9D 3H", "TH 5H QD 9S KS 7S 7C 6H 8C TD", "TH 2D 4D QC 5C 7D JD AH 9C 4H",
					"4H 3H AH 8D 6H QC QH 9H 2H 2C", "2D AD 4C TS 6H 7S TH 4H QS TD", "3C KD 2H 3H QS JD TC QC 5D 8H",
					"KS JC QD TH 9S KD 8D 8C 2D 9C", "3C QD KD 6D 4D 8D AH AD QC 8S", "8H 3S 9D 2S 3H KS 6H 4C 7C KC",
					"TH 9S 5C 3D 7D 6H AC 7S 4D 2C", "5C 3D JD 4D 2D 6D 5H 9H 4C KH", "AS 7H TD 6C 2H 3D QD KS 4C 4S",
					"JC 3C AC 7C JD JS 8H 9S QC 5D", "JD 6S 5S 2H AS 8C 7D 5H JH 3D", "8D TC 5S 9S 8S 3H JC 5H 7S AS",
					"5C TD 3D 7D 4H 8D 7H 4D 5D JS", "QS 9C KS TD 2S 8S 5C 2H 4H AS", "TH 7S 4H 7D 3H JD KD 5D 2S KC",
					"JD 7H 4S 8H 4C JS 6H QH 5S 4H", "2C QS 8C 5S 3H QC 2S 6C QD AD", "8C 3D JD TC 4H 2H AD 5S AC 2S",
					"5D 2C JS 2D AD 9D 3D 4C 4S JH", "8D 5H 5D 6H 7S 4D KS 9D TD JD", "3D 6D 9C 2S AS 7D 5S 5C 8H JD",
					"7C 8S 3S 6S 5H JD TC AD 7H 7S", "2S 9D TS 4D AC 8D 6C QD JD 3H", "9S KH 2C 3C AC 3D 5H 6H 8D 5D",
					"KS 3D 2D 6S AS 4C 2S 7C 7H KH", "AC 2H 3S JC 5C QH 4D 2D 5H 7S", "TS AS JD 8C 6H JC 8S 5S 2C 5D",
					"7S QH 7H 6C QC 8H 2D 7C JD 2S", "2C QD 2S 2H JC 9C 5D 2D JD JH", "7C 5C 9C 8S 7D 6D 8D 6C 9S JH",
					"2C AD 6S 5H 3S KS 7S 9D KH 4C", "7H 6C 2C 5C TH 9D 8D 3S QC AH", "5S KC 6H TC 5H 8S TH 6D 3C AH",
					"9C KD 4H AD TD 9S 4S 7D 6H 5D", "7H 5C 5H 6D AS 4C KD KH 4H 9D", "3C 2S 5C 6C JD QS 2H 9D 7D 3H",
					"AC 2S 6S 7S JS QD 5C QS 6H AD", "5H TH QC 7H TC 3S 7C 6D KC 3D", "4H 3D QC 9S 8H 2C 3S JC KS 5C",
					"4S 6S 2C 6H 8S 3S 3D 9H 3H JS", "4S 8C 4D 2D 8H 9H 7D 9D AH TS", "9S 2C 9H 4C 8D AS 7D 3D 6D 5S",
					"6S 4C 7H 8C 3H 5H JC AH 9D 9C", "2S 7C 5S JD 8C 3S 3D 4D 7D 6S", "3C KC 4S 5D 7D 3D JD 7H 3H 4H",
					"9C 9H 4H 4D TH 6D QD 8S 9S 7S", "2H AC 8S 4S AD 8C 2C AH 7D TC", "TS 9H 3C AD KS TC 3D 8C 8H JD",
					"QC 8D 2C 3C 7D 7C JD 9H 9C 6C", "AH 6S JS JH 5D AS QC 2C JD TD", "9H KD 2H 5D 2D 3S 7D TC AH TS",
					"TD 8H AS 5D AH QC AC 6S TC 5H", "KS 4S 7H 4D 8D 9C TC 2H 6H 3H", "3H KD 4S QD QH 3D 8H 8C TD 7S",
					"8S JD TC AH JS QS 2D KH KS 4D", "3C AD JC KD JS KH 4S TH 9H 2C", "QC 5S JS 9S KS AS 7C QD 2S JD",
					"KC 5S QS 3S 2D AC 5D 9H 8H KS", "6H 9C TC AD 2C 6D 5S JD 6C 7C", "QS KH TD QD 2C 3H 8S 2S QC AH",
					"9D 9H JH TC QH 3C 2S JS 5C 7H", "6C 3S 3D 2S 4S QD 2D TH 5D 2C", "2D 6H 6D 2S JC QH AS 7H 4H KH",
					"5H 6S KS AD TC TS 7C AC 4S 4H", "AD 3C 4H QS 8C 9D KS 2H 2D 4D", "4S 9D 6C 6D 9C AC 8D 3H 7H KD",
					"JC AH 6C TS JD 6D AD 3S 5D QD", "JC JH JD 3S 7S 8S JS QC 3H 4S", "JD TH 5C 2C AD JS 7H 9S 2H 7S",
					"8D 3S JH 4D QC AS JD 2C KC 6H", "2C AC 5H KD 5S 7H QD JH AH 2D", "JC QH 8D 8S TC 5H 5C AH 8C 6C",
					"3H JS 8S QD JH 3C 4H 6D 5C 3S", "6D 4S 4C AH 5H 5S 3H JD 7C 8D", "8H AH 2H 3H JS 3C 7D QC 4H KD",
					"6S 2H KD 5H 8H 2D 3C 8S 7S QD", "2S 7S KC QC AH TC QS 6D 4C 8D", "5S 9H 2C 3S QD 7S 6C 2H 7C 9D",
					"3C 6C 5C 5S JD JC KS 3S 5D TS", "7C KS 6S 5S 2S 2D TC 2H 5H QS", "AS 7H 6S TS 5H 9S 9D 3C KD 2H",
					"4S JS QS 3S 4H 7C 2S AC 6S 9D", "8C JH 2H 5H 7C 5D QH QS KH QC", "3S TD 3H 7C KC 8D 5H 8S KH 8C",
					"4H KH JD TS 3C 7H AS QC JS 5S", "AH 9D 2C 8D 4D 2D 6H 6C KC 6S", "2S 6H 9D 3S 7H 4D KH 8H KD 3D",
					"9C TC AC JH KH 4D JD 5H TD 3S", "7S 4H 9D AS 4C 7D QS 9S 2S KH", "3S 8D 8S KS 8C JC 5C KH 2H 5D",
					"8S QH 2C 4D KC JS QC 9D AC 6H", "8S 8C 7C JS JD 6S 4C 9C AC 4S", "QH 5D 2C 7D JC 8S 2D JS JH 4C",
					"JS 4C 7S TS JH KC KH 5H QD 4S", "QD 8C 8D 2D 6S TD 9D AC QH 5S", "QH QC JS 3D 3C 5C 4H KH 8S 7H",
					"7C 2C 5S JC 8S 3H QC 5D 2H KC", "5S 8D KD 6H 4H QD QH 6D AH 3D", "7S KS 6C 2S 4D AC QS 5H TS JD",
					"7C 2D TC 5D QS AC JS QC 6C KC", "2C KS 4D 3H TS 8S AD 4H 7S 9S", "QD 9H QH 5H 4H 4D KH 3S JC AD",
					"4D AC KC 8D 6D 4C 2D KH 2C JD", "2C 9H 2D AH 3H 6D 9C 7D TC KS", "8C 3H KD 7C 5C 2S 4S 5H AS AH",
					"TH JD 4H KD 3H TC 5C 3S AC KH", "6D 7H AH 7S QC 6H 2D TD JD AS", "JH 5D 7H TC 9S 7D JC AS 5S KH",
					"2H 8C AD TH 6H QD KD 9H 6S 6C", "QH KC 9D 4D 3S JS JH 4H 2C 9H", "TC 7H KH 4H JC 7D 9S 3H QS 7S",
					"AD 7D JH 6C 7H 4H 3S 3H 4D QH", "JD 2H 5C AS 6C QC 4D 3C TC JH", "AC JD 3H 6H 4C JC AD 7D 7H 9H",
					"4H TC TS 2C 8C 6S KS 2H JD 9S", "4C 3H QS QC 9S 9H 6D KC 9D 9C", "5C AD 8C 2C QH TH QD JC 8D 8H",
					"QC 2C 2S QD 9C 4D 3S 8D JH QS", "9D 3S 2C 7S 7C JC TD 3C TC 9H", "3C TS 8H 5C 4C 2C 6S 8D 7C 4H",
					"KS 7H 2H TC 4H 2C 3S AS AH QS", "8C 2D 2H 2C 4S 4C 6S 7D 5S 3S", "TH QC 5D TD 3C QS KD KC KS AS",
					"4D AH KD 9H KS 5C 4C 6H JC 7S", "KC 4H 5C QS TC 2H JC 9S AH QH", "4S 9H 3H 5H 3C QD 2H QC JH 8H",
					"5D AS 7H 2C 3D JH 6H 4C 6S 7D", "9C JD 9H AH JS 8S QH 3H KS 8H", "3S AC QC TS 4D AD 3D AH 8S 9H",
					"7H 3H QS 9C 9S 5H JH JS AH AC", "8D 3C JD 2H AC 9C 7H 5S 4D 8H", "7C JH 9H 6C JS 9S 7H 8C 9D 4H",
					"2D AS 9S 6H 4D JS JH 9H AD QD", "6H 7S JH KH AH 7H TD 5S 6S 2C", "8H JH 6S 5H 5S 9D TC 4C QC 9S",
					"7D 2C KD 3H 5H AS QD 7H JS 4D", "TS QH 6C 8H TH 5H 3C 3H 9C 9D", "AD KH JS 5D 3H AS AC 9S 5C KC",
					"2C KH 8C JC QS 6D AH 2D KC TC", "9D 3H 2S 7C 4D 6D KH KS 8D 7D", "9H 2S TC JH AC QC 3H 5S 3S 8H",
					"3S AS KD 8H 4C 3H 7C JH QH TS", "7S 6D 7H 9D JH 4C 3D 3S 6C AS", "4S 2H 2C 4C 8S 5H KC 8C QC QD",
					"3H 3S 6C QS QC 2D 6S 5D 2C 9D", "2H 8D JH 2S 3H 2D 6C 5C 7S AD", "9H JS 5D QH 8S TS 2H 7S 6S AD",
					"6D QC 9S 7H 5H 5C 7D KC JD 4H", "QC 5S 9H 9C 4D 6S KS 2S 4C 7C", "9H 7C 4H 8D 3S 6H 5C 8H JS 7S",
					"2D 6H JS TD 4H 4D JC TH 5H KC", "AC 7C 8D TH 3H 9S 2D 4C KC 4D", "KD QS 9C 7S 3D KS AD TS 4C 4H",
					"QH 9C 8H 2S 7D KS 7H 5D KD 4C", "9C 2S 2H JC 6S 6C TC QC JH 5C", "7S AC 8H KC 8S 6H QS JC 3D 6S",
					"JS 2D JH 8C 4S 6H 8H 6D 5D AD", "6H 7D 2S 4H 9H 7C AS AC 8H 5S", "3C JS 4S 6D 5H 2S QH 6S 9C 2C",
					"3D 5S 6S 9S 4C QS 8D QD 8S TC", "9C 3D AH 9H 5S 2C 7D AD JC 3S", "7H TC AS 3C 6S 6D 7S KH KC 9H",
					"3S TC 8H 6S 5H JH 8C 7D AC 2S", "QD 9D 9C 3S JC 8C KS 8H 5D 4D", "JS AH JD 6D 9D 8C 9H 9S 8H 3H",
					"2D 6S 4C 4D 8S AD 4S TC AH 9H", "TS AC QC TH KC 6D 4H 7S 8C 2H", "3C QD JS 9D 5S JC AH 2H TS 9H",
					"3H 4D QH 5D 9C 5H 7D 4S JC 3S", "8S TH 3H 7C 2H JD JS TS AC 8D", "9C 2H TD KC JD 2S 8C 5S AD 2C",
					"3D KD 7C 5H 4D QH QD TC 6H 7D", "7H 2C KC 5S KD 6H AH QC 7S QH", "6H 5C AC 5H 2C 9C 2D 7C TD 2S",
					"4D 9D AH 3D 7C JD 4H 8C 4C KS", "TH 3C JS QH 8H 4C AS 3D QS QC", "4D 7S 5H JH 6D 7D 6H JS KH 3C",
					"QD 8S 7D 2H 2C 7C JC 2S 5H 8C", "QH 8S 9D TC 2H AD 7C 8D QD 6S", "3S 7C AD 9H 2H 9S JD TS 4C 2D",
					"3S AS 4H QC 2C 8H 8S 7S TD TC", "JH TH TD 3S 4D 4H 5S 5D QS 2C", "8C QD QH TC 6D 4S 9S 9D 4H QC",
					"8C JS 9D 6H JD 3H AD 6S TD QC", "KC 8S 3D 7C TD 7D 8D 9H 4S 3S", "6C 4S 3D 9D KD TC KC KS AC 5S",
					"7C 6S QH 3D JS KD 6H 6D 2D 8C", "JD 2S 5S 4H 8S AC 2D 6S TS 5C", "5H 8C 5S 3C 4S 3D 7C 8D AS 3H",
					"AS TS 7C 3H AD 7D JC QS 6C 6H", "3S 9S 4C AC QH 5H 5D 9H TS 4H", "6C 5C 7H 7S TD AD JD 5S 2H 2S",
					"7D 6C KC 3S JD 8D 8S TS QS KH", "8S QS 8D 6C TH AC AH 2C 8H 9S", "7H TD KH QH 8S 3D 4D AH JD AS",
					"TS 3D 2H JC 2S JH KH 6C QC JS", "KC TH 2D 6H 7S 2S TC 8C 9D QS", "3C 9D 6S KH 8H 6D 5D TH 2C 2H",
					"6H TC 7D AD 4D 8S TS 9H TD 7S", "JS 6D JD JC 2H AC 6C 3D KH 8D", "KH JD 9S 5D 4H 4C 3H 7S QS 5C",
					"4H JD 5D 3S 3C 4D KH QH QS 7S", "JD TS 8S QD AH 4C 6H 3S 5S 2C", "QS 3D JD AS 8D TH 7C 6S QC KS",
					"7S 2H 8C QC 7H AC 6D 2D TH KH", "5S 6C 7H KH 7D AH 8C 5C 7S 3D", "3C KD AD 7D 6C 4D KS 2D 8C 4S",
					"7C 8D 5S 2D 2S AH AD 2C 9D TD", "3C AD 4S KS JH 7C 5C 8C 9C TH", "AS TD 4D 7C JD 8C QH 3C 5H 9S",
					"3H 9C 8S 9S 6S QD KS AH 5H JH", "QC 9C 5S 4H 2H TD 7D AS 8C 9D", "8C 2C 9D KD TC 7S 3D KH QC 3C",
					"4D AS 4C QS 5S 9D 6S JD QH KS", "6D AH 6C 4C 5H TS 9H 7D 3D 5S", "QS JD 7C 8D 9C AC 3S 6S 6C KH",
					"8H JH 5D 9S 6D AS 6S 3S QC 7H", "QD AD 5C JH 2H AH 4H AS KC 2C", "JH 9C 2C 6H 2D JS 5D 9H KC 6D",
					"7D 9D KD TH 3H AS 6S QC 6H AD", "JD 4H 7D KC 3H JS 3C TH 3D QS", "4C 3H 8C QD 5H 6H AS 8H AD JD",
					"TH 8S KD 5D QC 7D JS 5S 5H TS", "7D KC 9D QS 3H 3C 6D TS 7S AH", "7C 4H 7H AH QC AC 4D 5D 6D TH",
					"3C 4H 2S KD 8H 5H JH TC 6C JD", "4S 8C 3D 4H JS TD 7S JH QS KD", "7C QC KD 4D 7H 6S AD TD TC KH",
					"5H 9H KC 3H 4D 3D AD 6S QD 6H", "TH 7C 6H TS QH 5S 2C KC TD 6S", "7C 4D 5S JD JH 7D AC KD KH 4H",
					"7D 6C 8D 8H 5C JH 8S QD TH JD", "8D 7D 6C 7C 9D KD AS 5C QH JH", "9S 2C 8C 3C 4C KS JH 2D 8D 4H",
					"7S 6C JH KH 8H 3H 9D 2D AH 6D", "4D TC 9C 8D 7H TD KS TH KD 3C", "JD 9H 8D QD AS KD 9D 2C 2S 9C",
					"8D 3H 5C 7H KS 5H QH 2D 8C 9H", "2D TH 6D QD 6C KC 3H 3S AD 4C", "4H 3H JS 9D 3C TC 5H QH QC JC",
					"3D 5C 6H 3S 3C JC 5S 7S 2S QH", "AC 5C 8C 4D 5D 4H 2S QD 3C 3H", "2C TD AH 9C KD JS 6S QD 4C QC",
					"QS 8C 3S 4H TC JS 3H 7C JC AD", "5H 4D 9C KS JC TD 9S TS 8S 9H", "QD TS 7D AS AC 2C TD 6H 8H AH",
					"6S AD 8C 4S 9H 8D 9D KH 8S 3C", "QS 4D 2D 7S KH JS JC AD 4C 3C", "QS 9S 7H KC TD TH 5H JS AC JH",
					"6D AC 2S QS 7C AS KS 6S KH 5S", "6D 8H KH 3C QS 2H 5C 9C 9D 6C", "JS 2C 4C 6H 7D JC AC QD TD 3H",
					"4H QC 8H JD 4C KD KS 5C KC 7S", "6D 2D 3H 2S QD 5S 7H AS TH 6S", "AS 6D 8D 2C 8S TD 8H QD JC AH",
					"9C 9H 2D TD QH 2H 5C TC 3D 8H", "KC 8S 3D KH 2S TS TC 6S 4D JH", "9H 9D QS AC KC 6H 5D 4D 8D AH",
					"9S 5C QS 4H 7C 7D 2H 8S AD JS", "3D AC 9S AS 2C 2D 2H 3H JC KH", "7H QH KH JD TC KS 5S 8H 4C 8D",
					"2H 7H 3S 2S 5H QS 3C AS 9H KD", "AD 3D JD 6H 5S 9C 6D AC 9S 3S", "3D 5D 9C 2D AC 4S 2S AD 6C 6S",
					"QC 4C 2D 3H 6S KC QH QD 2H JH", "QC 3C 8S 4D 9S 2H 5C 8H QS QD", "6D KD 6S 7H 3S KH 2H 5C JC 6C",
					"3S 9S TC 6S 8H 2D AD 7S 8S TS", "3C 6H 9C 3H 5C JC 8H QH TD QD", "3C JS QD 5D TD 2C KH 9H TH AS",
					"9S TC JD 3D 5C 5H AD QH 9H KC", "TC 7H 4H 8H 3H TD 6S AC 7C 2S", "QS 9D 5D 3C JC KS 4D 6C JH 2S",
					"9S 6S 3C 7H TS 4C KD 6D 3D 9C", "2D 9H AH AC 7H 2S JH 3S 7C QC", "QD 9H 3C 2H AC AS 8S KD 8C KH",
					"2D 7S TD TH 6D JD 8D 4D 2H 5S", "8S QH KD JD QS JH 4D KC 5H 3S", "3C KH QC 6D 8H 3S AH 7D TD 2D",
					"5S 9H QH 4S 6S 6C 6D TS TH 7S", "6C 4C 6D QS JS 9C TS 3H 8D 8S", "JS 5C 7S AS 2C AH 2H AD 5S TC",
					"KD 6C 9C 9D TS 2S JC 4H 2C QD", "QS 9H TC 3H KC KS 4H 3C AD TH", "KH 9C 2H KD 9D TC 7S KC JH 2D",
					"7C 3S KC AS 8C 5D 9C 9S QH 3H", "2D 8C TD 4C 2H QC 5D TC 2C 7D", "KS 4D 6C QH TD KH 5D 7C AD 8D",
					"2S 9S 8S 4C 8C 3D 6H QD 7C 7H", "6C 8S QH 5H TS 5C 3C 4S 2S 2H", "8S 6S 2H JC 3S 3H 9D 8C 2S 7H",
					"QC 2C 8H 9C AC JD 4C 4H 6S 3S", "3H 3S 7D 4C 9S 5H 8H JC 3D TC", "QH 2S 2D 9S KD QD 9H AD 6D 9C",
					"8D 2D KS 9S JC 4C JD KC 4S TH", "KH TS 6D 4D 5C KD 5H AS 9H AD", "QD JS 7C 6D 5D 5C TH 5H QH QS",
					"9D QH KH 5H JH 4C 4D TC TH 6C", "KH AS TS 9D KD 9C 7S 4D 8H 5S", "KH AS 2S 7D 9D 4C TS TH AH 7C",
					"KS 4D AC 8S 9S 8D TH QH 9D 5C", "5D 5C 8C QS TC 4C 3D 3S 2C 8D", "9D KS 2D 3C KC 4S 8C KH 6C JC",
					"8H AH 6H 7D 7S QD 3C 4C 6C KC", "3H 2C QH 8H AS 7D 4C 8C 4H KC", "QD 5S 4H 2C TD AH JH QH 4C 8S",
					"3H QS 5S JS 8H 2S 9H 9C 3S 2C", "6H TS 7S JC QD AC TD KC 5S 3H", "QH AS QS 7D JC KC 2C 4C 5C 5S",
					"QH 3D AS JS 4H 8D 7H JC 2S 9C", "5D 4D 2S 4S 9D 9C 2D QS 8H 7H", "6D 7H 3H JS TS AC 2D JH 7C 8S",
					"JH 5H KC 3C TC 5S 9H 4C 8H 9D", "8S KC 5H 9H AD KS 9D KH 8D AH", "JC 2H 9H KS 6S 3H QC 5H AH 9C",
					"5C KH 5S AD 6C JC 9H QC 9C TD", "5S 5D JC QH 2D KS 8H QS 2H TS", "JH 5H 5S AH 7H 3C 8S AS TD KH",
					"6H 3D JD 2C 4C KC 7S AH 6C JH", "4C KS 9D AD 7S KC 7D 8H 3S 9C", "7H 5C 5H 3C 8H QC 3D KH 6D JC",
					"2D 4H 5D 7D QC AD AH 9H QH 8H", "KD 8C JS 9D 3S 3C 2H 5D 6D 2S", "8S 6S TS 3C 6H 8D 5S 3H TD 6C",
					"KS 3D JH 9C 7C 9S QS 5S 4H 6H", "7S 6S TH 4S KC KD 3S JC JH KS", "7C 3C 2S 6D QH 2C 7S 5H 8H AH",
					"KC 8D QD 6D KH 5C 7H 9D 3D 9C", "6H 2D 8S JS 9S 2S 6D KC 7C TC", "KD 9C JH 7H KC 8S 2S 7S 3D 6H",
					"4H 9H 2D 4C 8H 7H 5S 8S 2H 8D", "AD 7C 3C 7S 5S 4D 9H 3D JC KH", "5D AS 7D 6D 9C JC 4C QH QS KH",
					"KD JD 7D 3D QS QC 8S 6D JS QD", "6S 8C 5S QH TH 9H AS AC 2C JD", "QC KS QH 7S 3C 4C 5C KC 5D AH",
					"6C 4H 9D AH 2C 3H KD 3D TS 5C", "TD 8S QS AS JS 3H KD AC 4H KS", "7D 5D TS 9H 4H 4C 9C 2H 8C QC",
					"2C 7D 9H 4D KS 4C QH AD KD JS", "QD AD AH KH 9D JS 9H JC KD JD", "8S 3C 4S TS 7S 4D 5C 2S 6H 7C",
					"JS 7S 5C KD 6D QH 8S TD 2H 6S", "QH 6C TC 6H TD 4C 9D 2H QC 8H", "3D TS 4D 2H 6H 6S 2C 7H 8S 6C",
					"9H 9D JD JH 3S AH 2C 6S 3H 8S", "2C QS 8C 5S 3H 2S 7D 3C AD 4S", "5C QC QH AS TS 4S 6S 4C 5H JS",
					"JH 5C TD 4C 6H JS KD KH QS 4H", "TC KH JC 4D 9H 9D 8D KC 3C 8H", "2H TC 8S AD 9S 4H TS 7H 2C 5C",
					"4H 2S 6C 5S KS AH 9C 7C 8H KD", "TS QH TD QS 3C JH AH 2C 8D 7D", "5D KC 3H 5S AC 4S 7H QS 4C 2H",
					"3D 7D QC KH JH 6D 6C TD TH KD", "5S 8D TH 6C 9D 7D KH 8C 9S 6D", "JD QS 7S QC 2S QH JC 4S KS 8D",
					"7S 5S 9S JD KD 9C JC AD 2D 7C", "4S 5H AH JH 9C 5D TD 7C 2D 6S", "KC 6C 7H 6S 9C QD 5S 4H KS TD",
					"6S 8D KS 2D TH TD 9H JD TS 3S", "KH JS 4H 5D 9D TC TD QC JD TS", "QS QD AC AD 4C 6S 2D AS 3H KC",
					"4C 7C 3C TD QS 9C KC AS 8D AD", "KC 7H QC 6D 8H 6S 5S AH 7S 8C", "3S AD 9H JC 6D JD AS KH 6S JH",
					"AD 3D TS KS 7H JH 2D JS QD AC", "9C JD 7C 6D TC 6H 6C JC 3D 3S", "QC KC 3S JC KD 2C 8D AH QS TS",
					"AS KD 3D JD 8H 7C 8C 5C QD 6C"]
	
	def analyzeHand(hand):
		# Convert cards into pairs of numbers and suits for ease of comparison.
		suits = []
		values = []
		
		for card in hand:
		
			value = card[0]
			if value == 'T': value = 10
			elif value == 'J': value = 11
			elif value == 'Q': value = 12
			elif value == 'K': value = 13
			elif value == 'A': value = 14
			else: value = int(value)
			
			suit = ord(card[1]) # each letter is different, but we only care about similarities and not suit value.
			
			values.append(value)
			suits.append(suit)
			
		suits.sort()
		values.sort()
		
		attributes = dict()
		
		# Make comparisons such that all equal terms always can be compared to each other, and contain enough information to
		# sort themselves if they need to go back to highest card values in ties.		
		
		if suits.count(suits[0]) == len(suits):
			attributes['FLUSH'] = values[::-1]
			
		if not False in [values[i] - values[0] == i for i in range(5)]:
			attributes['STRAIGHT'] = values[::-1]
			
		if "STRAIGHT" in attributes and "FLUSH" in attributes:
			attributes['STRAIGHT FLUSH'] = values[::-1]
			
		if len(attributes) == 0:
			# Then we have mixed suits and values, and only need to look for repeats or the highest card.
			# Check each value once to see how many repeats exist.
						
			for term in set(values):
				count = values.count(term)
				
				if count == 2:
					# Find the terms not in the pair.
					pairTerms = set(values)
					pairTerms.remove(term)
					
					# If we have two pair, we need to also remove the existing pair from the non-pair terms, in case it is
					# higher than this pair.
					if 'PAIR' in attributes: pairTerms.remove(attributes['PAIR'][0])
					
					# Order the terms for comparison.
					pairTerms = list(pairTerms)
					pairTerms.sort()
					
					# Reverse them so that largest is first.
					pairTerms = pairTerms[::-1]
					
					# Put the pair in front.
					
					if 'PAIR' in attributes:
						pairTerms = [max(term,attributes['PAIR'][0])] + [min(term, attributes['PAIR'][0])] + pairTerms
						attributes['TWOPAIR'] = pairTerms	
					else:
						pairTerms = [term] + pairTerms
						attributes['PAIR'] = pairTerms
					
					# Now we can compare arbitrary pairs with each other.									
					
				elif count > 2:
						pairTerms = set(values)
						pairTerms.remove(term)
						pairTerms = list(pairTerms)
						pairTerms.sort()
						pairTerms = pairTerms[::-1]
						pairTerms = [term] + pairTerms
						
						if count == 3:
							attributes['THREEOFAKIND'] = pairTerms
						elif count == 4:
							attributes['FOUROFAKIND'] = pairTerms
						
					
			if 'PAIR' in attributes and 'THREEOFAKIND' in attributes:					
				attributes['FULLHOUSE'] = [attributes['THREEOFAKIND'][0], attributes['PAIR'][0]]
					
		# Just in case the hand sucks.
		attributes["HIGH"] = values[::-1]
		return attributes
		
		
	def whoWins(term, player_a, player_b):
		if term in player_a and not term in player_b: return 1
		if term in player_b and not term in player_a: return 0
		
		if term in player_a and term in player_b:
			if player_a[term] > player_b[term]: return 1
			return 0	
		
		return None
	
	def rateHands(hand_a, hand_b):
	
		# Quick things!
		a = analyzeHand(hand_a)
		b = analyzeHand(hand_b)
		
		# 1 indicates hand a wins, 0 indicates hand b wins.
		
		# Do some super hax checks to see if one player is definitly ahead of the other.  This should
		# speed up any cases where one player obviously wins.
		if len(a) == 1 and len(b) > 1: return 0
		if len(b) == 1 and len(a) > 1: return 1		
		
		test_sf = whoWins('STRAIGHT FLUSH', a, b)
		if test_sf is not None: return test_sf
			
		test_4k = whoWins('FOUROFAKIND', a, b)
		if test_4k is not None: return test_4k
		
		test_fh = whoWins('FULLHOUSE', a, b)
		if test_fh is not None: return test_fh
		
		test_fl = whoWins('FLUSH', a, b)
		if test_fl is not None: return test_fl
		
		test_st = whoWins('STRAIGHT', a, b)
		if test_st is not None: return test_st
		
		test_3k = whoWins('THREEOFAKIND', a, b)
		if test_3k is not None: return test_3k
		
		test_2p = whoWins('TWOPAIR', a, b)
		if test_2p is not None: return test_2p
			
		test_1p = whoWins('PAIR', a, b)
		if test_1p is not None: return test_1p

		return whoWins('HIGH', a, b)

	wins = 0
	for cards in pokerstrings:
		allcards = cards.split(' ')
		player1 = allcards[:5]
		player2 = allcards[5:]
		wins += rateHands(player1, player2)
		
	return wins

# -----------------------------
# -----------------------------

@SetupAndTime(55,'Number of Lychrel/invalid numbers found up to %(upperLimit):', upperLimit = 10000)
def problem55(**kwargs):		

	upperLimit = kwargs['upperLimit']
	
	invalidTerms = []
	
	for i in xrange(upperLimit):
	
		# Assumptions:
		# - (i) (the number will) become a palindrome in less than fifty iterations, or,
		# - (ii) no one, with all the computing power that exists, has managed so far to map it to a palindrome.
	
		term = i
		valid = False
	
		for j in xrange(50):
			# Always add reverse of number to itself.
			term += int(str(term)[::-1])

			# If the number generated is palindromic, we're good.
			if int(str(term)[::-1]) == term:
				valid = True
				break
				
		if not valid:
			invalidTerms.append(i)
				
	return str(len(invalidTerms)) + '\n\n' + str(invalidTerms)

# -----------------------------
# -----------------------------

@SetupAndTime(56,'Maximum digit sum for a**b for a<=%(maxv), b<=%(maxv):', maxv = 100)
def problem56(**kwargs):
	
	maxnum = kwargs['maxv']
	maxsum = [0, 0, 0]
	
	for a in xrange(1,maxnum+1):
		for b in xrange(1,maxnum+1):
			abstar = a**b
			total = sum([int(i) for i in str(abstar)])
			if total > maxsum[0]:
				maxsum = [total, a, b]
				
	return stringify('a =', maxsum[1], '\nb =', maxsum[2])

# -----------------------------
# -----------------------------

@SetupAndTime(57,'In the first %(maxv) expansions of sqrt(%(toExpand)) fraction count with more digits in numerator?', maxv = 1000, toExpand = 2)
def problem57(**kwargs):
	
	def squareRootExpand(n, totalrange):

		num = 1
		denom = 1
		
		while num*num < n: num += 1	
		num -= 1
		
		if num*num == n:
			yield num, denom
			return
		
		denom = n
		num = denom*num + 1
		
		yield num, denom
		
		iterOn = 1
		
		while 1:
			num, denom = (2*denom)+num, num+denom
			yield num, denom
			
			if iterOn > totalrange: break
			else: iterOn += 1
			

	numExpansions = kwargs['maxv']
	numToExpand = kwargs['toExpand']
	validTerms = []
	
	for num, denom in squareRootExpand(numToExpand, numExpansions):		
		if len(str(num)) > len(str(denom)):
			validTerms.append([num, denom])
		
	return  len(validTerms)

# -----------------------------
# -----------------------------

@SetupAndTime(58,'For a spiral grid, length of sides when the number of primes in edges drops below %(pct) percent:', pct=10)
def problem58(**kwargs):
	
	skip = 2
	lastSeen = 9
	
	primes = 3
	notprimes = 2
	
	pct = kwargs['pct']/100
	
	while primes*pct > primes+notprimes:

		skip += 2
		for i in xrange(4):
			lastSeen += skip
			
			if Prime.isPrime(lastSeen): 
				primes += 1	
			else: 
				notprimes += 1
				
	return skip+1

# -----------------------------
# -----------------------------

@SetupAndTime(59,'The %(keylen) letter key used to xor the message:', keylen=3)
def problem59(**kwargs):
	bytes = [79,59,12,2,79,35,8,28,20,2,3,68,8,9,68,45,0,12,9,67,68,4,7,5,23,27,1,21,79,85,78,79,85,71,38,10,71,27,12,2,79,6,2,8,13,9,1,13,9,8,68,19,7,1,71,56,11,21,11,68,6,3,22,2,14,0,30,79,1,31,6,23,19,10,0,73,79,44,2,79,19,6,28,68,16,6,16,15,79,35,8,11,72,71,14,10,3,79,12,2,79,19,6,28,68,32,0,0,73,79,86,71,39,1,71,24,5,20,79,13,9,79,16,15,10,68,5,10,3,14,1,10,14,1,3,71,24,13,19,7,68,32,0,0,73,79,87,71,39,1,71,12,22,2,14,16,2,11,68,2,25,1,21,22,16,15,6,10,0,79,16,15,10,22,2,79,13,20,65,68,41,0,16,15,6,10,0,79,1,31,6,23,19,28,68,19,7,5,19,79,12,2,79,0,14,11,10,64,27,68,10,14,15,2,65,68,83,79,40,14,9,1,71,6,16,20,10,8,1,79,19,6,28,68,14,1,68,15,6,9,75,79,5,9,11,68,19,7,13,20,79,8,14,9,1,71,8,13,17,10,23,71,3,13,0,7,16,71,27,11,71,10,18,2,29,29,8,1,1,73,79,81,71,59,12,2,79,8,14,8,12,19,79,23,15,6,10,2,28,68,19,7,22,8,26,3,15,79,16,15,10,68,3,14,22,12,1,1,20,28,72,71,14,10,3,79,16,15,10,68,3,14,22,12,1,1,20,28,68,4,14,10,71,1,1,17,10,22,71,10,28,19,6,10,0,26,13,20,7,68,14,27,74,71,89,68,32,0,0,71,28,1,9,27,68,45,0,12,9,79,16,15,10,68,37,14,20,19,6,23,19,79,83,71,27,11,71,27,1,11,3,68,2,25,1,21,22,11,9,10,68,6,13,11,18,27,68,19,7,1,71,3,13,0,7,16,71,28,11,71,27,12,6,27,68,2,25,1,21,22,11,9,10,68,10,6,3,15,27,68,5,10,8,14,10,18,2,79,6,2,12,5,18,28,1,71,0,2,71,7,13,20,79,16,2,28,16,14,2,11,9,22,74,71,87,68,45,0,12,9,79,12,14,2,23,2,3,2,71,24,5,20,79,10,8,27,68,19,7,1,71,3,13,0,7,16,92,79,12,2,79,19,6,28,68,8,1,8,30,79,5,71,24,13,19,1,1,20,28,68,19,0,68,19,7,1,71,3,13,0,7,16,73,79,93,71,59,12,2,79,11,9,10,68,16,7,11,71,6,23,71,27,12,2,79,16,21,26,1,71,3,13,0,7,16,75,79,19,15,0,68,0,6,18,2,28,68,11,6,3,15,27,68,19,0,68,2,25,1,21,22,11,9,10,72,71,24,5,20,79,3,8,6,10,0,79,16,8,79,7,8,2,1,71,6,10,19,0,68,19,7,1,71,24,11,21,3,0,73,79,85,87,79,38,18,27,68,6,3,16,15,0,17,0,7,68,19,7,1,71,24,11,21,3,0,71,24,5,20,79,9,6,11,1,71,27,12,21,0,17,0,7,68,15,6,9,75,79,16,15,10,68,16,0,22,11,11,68,3,6,0,9,72,16,71,29,1,4,0,3,9,6,30,2,79,12,14,2,68,16,7,1,9,79,12,2,79,7,6,2,1,73,79,85,86,79,33,17,10,10,71,6,10,71,7,13,20,79,11,16,1,68,11,14,10,3,79,5,9,11,68,6,2,11,9,8,68,15,6,23,71,0,19,9,79,20,2,0,20,11,10,72,71,7,1,71,24,5,20,79,10,8,27,68,6,12,7,2,31,16,2,11,74,71,94,86,71,45,17,19,79,16,8,79,5,11,3,68,16,7,11,71,13,1,11,6,1,17,10,0,71,7,13,10,79,5,9,11,68,6,12,7,2,31,16,2,11,68,15,6,9,75,79,12,2,79,3,6,25,1,71,27,12,2,79,22,14,8,12,19,79,16,8,79,6,2,12,11,10,10,68,4,7,13,11,11,22,2,1,68,8,9,68,32,0,0,73,79,85,84,79,48,15,10,29,71,14,22,2,79,22,2,13,11,21,1,69,71,59,12,14,28,68,14,28,68,9,0,16,71,14,68,23,7,29,20,6,7,6,3,68,5,6,22,19,7,68,21,10,23,18,3,16,14,1,3,71,9,22,8,2,68,15,26,9,6,1,68,23,14,23,20,6,11,9,79,11,21,79,20,11,14,10,75,79,16,15,6,23,71,29,1,5,6,22,19,7,68,4,0,9,2,28,68,1,29,11,10,79,35,8,11,74,86,91,68,52,0,68,19,7,1,71,56,11,21,11,68,5,10,7,6,2,1,71,7,17,10,14,10,71,14,10,3,79,8,14,25,1,3,79,12,2,29,1,71,0,10,71,10,5,21,27,12,71,14,9,8,1,3,71,26,23,73,79,44,2,79,19,6,28,68,1,26,8,11,79,11,1,79,17,9,9,5,14,3,13,9,8,68,11,0,18,2,79,5,9,11,68,1,14,13,19,7,2,18,3,10,2,28,23,73,79,37,9,11,68,16,10,68,15,14,18,2,79,23,2,10,10,71,7,13,20,79,3,11,0,22,30,67,68,19,7,1,71,8,8,8,29,29,71,0,2,71,27,12,2,79,11,9,3,29,71,60,11,9,79,11,1,79,16,15,10,68,33,14,16,15,10,22,73]
	
	common_threshold = 0.33
	uncommon_threshold = 0.01
		
	common_set = frozenset('aeiou tnsh')
	uncommon_set = frozenset([chr(x) for x in xrange(255)]) - set('1234567890 qwertyuiopasdfghjklzxcvbnm QWERTYUIOPASDFGHJKLZXCVBNM ,./:;\'"()')
	
	buckets = [[byte for byte in bytes[n::kwargs['keylen']]] for n in range(kwargs['keylen'])]
	chars = [set() for x in range(len(buckets))]
	
	# Try to see what has the most recurring characters.
	
	for char in [chr(x) for x in xrange(0, 255)]:
		for i, bucket in enumerate(buckets):
		
			# Keep track of what we've parsed to so that we can abort early if it seems to be a bunch of junk.
			common = 0
			uncommon = 0

			uncommon_stop_parsing = uncommon_threshold * len(bucket)
			common_reasonable_parse = common_threshold * len(bucket)
		
			for encodedChar in bucket:
				newchar = chr(ord(char)^encodedChar)
				
				if newchar in uncommon_set:
					uncommon += 1
					if uncommon >= uncommon_stop_parsing:
						break
						
				elif newchar.lower() in common_set:
					common += 1
						
			if uncommon < uncommon_stop_parsing and common >= common_reasonable_parse:
				chars[i].add(char)

	# Do permutations of sequential input.  Assuming I have [[a, b, c], [d, e, f], [g, h, i]],
	# return adg, adh, adi, aeg, aeh, aei, ... cfi
	
	def SetYieldCombination(l):
	
		# Base case, list has one element in it (input = [[a, b, c]])
		if len(l) == 1: 
			for elem in l[0]: 
				yield elem
			
		# Otherwise it has multiple sets we should split apart (input=[[a, b, c], [d, e, f]])
		else:			
			for remainder in SetYieldCombination(l[1:]):
				for term in l[0]:
					yield term + remainder
							
	for permutation in SetYieldCombination(chars):
	
		permutationStr = permutation*((len(bytes)/3)+1)
		permutationStr = permutationStr[0:len(bytes)]
		
		decoded = ''.join([chr(ord(p)^q) for p, q in zip(permutationStr, bytes)])
		
		hasWord = False
		
		# "the knowledge that the plain text must contain common English words"	
		for term in ['the', 'and', 'to', 'if', 'was', 'is']:
			if term in decoded:
				return "KEY: " + permutation + "\n" + decoded
						
	print 'Out of the possible key pairings', chars, '\nNo combinations yielded reasonable English text.'

# -----------------------------
# -----------------------------

@SetupAndTime(60,'The first set of %(numTerms) primes that always concat to a prime includes:',numTerms=5)
def problem60(**kwargs):
	# Go through a list of primes - find prime substrings and see if moving them to the back creates a prime.
	# Record these numbers and go from there.

	def doesConcat(a, b):
	
		stra, strb = str(a), str(b)
		ab = int(stra+strb)
		ba = int(strb+stra)
		
		abPrime, baPrime = False, False
		
		if ab < Prime.ordered[-1]:  abPrime = Prime.isPrime(ab)
		else:                       abPrime = Prime.isPrime(ab)			
		
		if not abPrime: return False


		if ba < Prime.ordered[-1]: baPrime = Prime.isPrime(ba)
		else:                      baPrime = Prime.isPrime(ba)
			
		if not baPrime: return False
			
		return True
				
				
	def checkMatches(mostRecent, container, minTerms):
			
		# Assume that the most recently added term must be in the group.
		if len(container[mostRecent])  < minTerms: return None

		allContaining = []
		intersections = []
		
		# See if there are any commonly used elements that must be in the final set given the
		# number of times they occur.
		freq = dict()
	
		# See what elements contain the most recent term.
		for element in container:
		
			# Retroactively put elements from the most recent term's pool into the frequency list too.
			if element in container[mostRecent]:  
				if element in freq: freq[element] += 1
				else:               freq[element] = 1
			
			# See if any sets containing the most recent elements work with the correct number of terms.
			if mostRecent in container[element] and len(container[element]) >= minTerms:

				testSet = set(container[mostRecent])
				testAgainst = set(container[element])				
				
				testSet.add(mostRecent)
				testAgainst.add(element)
								
				intersection = testSet.intersection(testAgainst)
				
				if len(intersection) >= minTerms:
					allContaining.append(element)
					intersections.append(intersection)
					
					# Only increase frequency of numbers if we're using a term that could be valid.
					for testElem in testAgainst:
					
						if testElem not in freq: freq[testElem] = 1
						else:                    freq[testElem] += 1
					
		# We need to have a certain number of terms that mingle with each other correctly.
		# What terms are repeated enough that all set must contain at least some number of them?
		mustContain = []
		
		for element in freq:
			if freq[element] >= minTerms: mustContain.append(element)
		
		# What elements, containing the most recent element, also contain the minimum number of
		# terms present in at least minTerms sets?
		
		finalContainingList = []
		
		if len(mustContain) == minTerms:
			finalContainingList = mustContain
			
		elif len(mustContain) > minTerms:
			for element in allContaining:
				numShared = 0
				
				for musthave in mustContain:
					if musthave in container[element]: numShared += 1
						
				if numShared >= minTerms: finalContainingList.append(element)
					
		else:
			return None
			
		# This is probably a horrible way to do these permutations.  While I could probably come up with a
		# smarter way to handle the "test and narrow down the combinations of the remaining terms."
		# I'm going to generate lists of indices in the final list that need to be tested.
		# Probably can just generate a list using the raw terms themselves, I don't see why I'm exactly
		# trying to do an index based solution for it.
		
		# At least all the filters above narrow down the number of terms enough so that this runs quickly for n=5...
		
		idxTests = [x for x in itertools.permutations(range(len(finalContainingList)), minTerms-1)]
		valids = []
		
		for test in idxTests:
		
			testSet = set(container[mostRecent])
			testSet.add(mostRecent)
			
			for element in test:
				filterSet = set(container[finalContainingList[element]])
				filterSet.add(finalContainingList[element])
			
				testSet = testSet.intersection(filterSet)
				if len(testSet) < minTerms:
					break
				
			if len(testSet) >= minTerms:
				validTerms = list(testSet)
				validTerms.sort()
				if not validTerms in valids:
					valids.append(validTerms)

		if len(valids) == 0: return None
		
		# Return the thing with the lowest sum, in case multiple combinations of terms using this number all
		# fit the concat/prime criteria.
		
		sums = [sum(t) for t in valids]
		smallest = min(sums)
		idx = sums.index(smallest)
		
		return valids[0]
		
	numTerms = kwargs['numTerms']
	
	validTerms = []
	goesWith = dict()
	oldPrimes = [Prime.ordered[0]]
	
	for prime in Prime.ordered[1:]:
	
		# Assume the concat check doesn't work.
		somethingChanged = False
		goesWith[prime] = set()
		
		for prime2 in oldPrimes:
			if prime != prime2:
				if doesConcat(prime, prime2):
					somethingChanged = True
					goesWith[prime].add(prime2)
					goesWith[prime2].add(prime)
		
		if somethingChanged:
			result = checkMatches(prime, goesWith, numTerms)
					
			if result is not None:
				validTerms = result
				break	
				
		oldPrimes.append(prime)
		
	return validTerms

# -----------------------------
# -----------------------------

@SetupAndTime(61,'Set of 6 numbers that loop and contain different terms:')
def problem61(**kwargs):

	def triangle(n): return n * (n + 1) / 2
	def square(n):   return n * (n  )
	def pentagon(n): return n * (3 * n - 1) / 2
	def hexagon(n):  return n * (2 * n - 1)
	def heptagon(n): return n * (5 * n - 3) / 2
	def octagon(n):  return n * (3 * n - 2)
		
	def tryToConnect(path, alreadyUsed, allTerms, numTerms):
		
		if len(alreadyUsed) == len(allTerms): return None		
		remainingTerms = len(allTerms) - len(path)
		
		# Base case, if this is the final result, what is the transition between the prior term and first term
		# that bust be in the list of every term?
		
		mustInclude = int(str(path[-1])[2:] + str(path[0])[:2])
		
		# This just tells us the index of the array we're looking at.
		for pos in xrange(len(allTerms)):
		
			if pos in alreadyUsed: continue
			
			# Check to see if any terms in this element's list directly match.
			if mustInclude in allTerms[pos] and remainingTerms == 1: return [mustInclude]
					
			else:
			
				# Find all transition numbers that could work.
				candidates = []
				
				for number in allTerms[pos]:
					# Does the end of the last number match the beginning of this number?
					if str(path[-1])[2:] == str(number)[:2]: candidates.append(number)
				
				# Assuming we may have multiple terms that fit the criteria...which are valid?
				results = []
				
				for number in candidates:
					result = tryToConnect(path + [number], [pos] + alreadyUsed, allTerms, numTerms)
					
					if result is not None:
						expanded = path + [number] + result
						
						if len(expanded) == numTerms:
							results.append([number] + result)
				
				if len(results) > 0:
					sums = [sum(x) for x in results]
					return results[sums.index(min(sums))]
						
		return None

	
	# No generalizations for this because I'
	
	numDigitsInTerm = 4
	
	maxterm = maxDigitVal(numDigitsInTerm)
	minterm = int(str(maxterm)[1:]) + 1
	
	termsInRange = []
	
	#tries = [triangle, square, pentagon]
	tries = [triangle, square, pentagon, hexagon, heptagon, octagon]
	
	for shapefunc in tries:
	
		terms = []
		counter = 1
		
		while len(terms) == 0 or terms[-1] <= maxterm:
			nextTerm = shapefunc(counter)
			if nextTerm > maxterm: break
			if minterm <= nextTerm <= maxterm: terms.append(nextTerm)
			counter += 1
		
		termsInRange.append(terms)
		
	
	# Just brute force everything.  As one number from each shape type needs to exist,
	# we can start with any one of them and try to connect the remaining numbers.
	for element in termsInRange[-1]:
		used = [len(termsInRange)-1]
		path = [element]
		result = tryToConnect(path, used, termsInRange, len(tries))
		if result is not None:
			print 'Set of %(tries) numbers that loop and contain different terms:'
			endlist = [element] + result
			return stringify(endlist, '\n\n', sum(endlist))

# -----------------------------
# -----------------------------

@SetupAndTime(62,'The first set of %(targetNumTerms) terms whose cubes are permutations of each other:', targetNumTerms=5)
def problem62(**kwargs):
	
	# Sort terms by a dict of repeated numbers.
	foundTerms = dict()

	targetNumTerms = kwargs['targetNumTerms']
		
	found = False
	counter = 1
	while not found:
		cubeTerm = counter**3
		strTerm = str(cubeTerm)
		numDigits = len(strTerm)
		
		if not numDigits in foundTerms:
			foundTerms[numDigits] = dict()
			for i in [str(z) for z in xrange(10)]:
				foundTerms[numDigits][i] = dict()
			
		# The initial intersection set will be the set of terms containing the first digit
		# the same number of times it occurs in the term we're looking at.
		
		initialCount = strTerm.count(strTerm[0])
		initialSet = set()
		
		# The set doesn't contain the terms we're adding yet, but to do one loop instead of two,
		# we'll do intersections on this set after appending our term to the other terms with this criteria,
		# and the number of total matches in the end will be the result of those intersections.
		
		if strTerm[0] in foundTerms[numDigits] and initialCount in foundTerms[numDigits][strTerm[0]]:
			initialSet = foundTerms[numDigits][strTerm[0]][initialCount]
			initialSet.add(counter)
		
		for eachchar in strTerm:
		
			timesPresent = strTerm.count(eachchar)
		
			# Three dimensional dictionaries probably are a bit gross, but as a reminder:
			# - The overarching dict (foundTerms[n]) finds terms with n digits.
			# - The second layer     (foundTerms[x][n]) finds x length terms with the digit n.
			# - The third layer      (foundTerms[x][y][n]) finds x length terms, with the digit y, repeated n times in the cubed term.
			
			if timesPresent not in foundTerms[numDigits][eachchar]:
				foundTerms[numDigits][eachchar][timesPresent] = set()
			
			foundTerms[numDigits][eachchar][timesPresent].add(counter)
			initialSet = initialSet.intersection(foundTerms[numDigits][eachchar][timesPresent])
			
		if len(initialSet) >= targetNumTerms:
			found = True			
			return stringify(EXPAND([stringify('\t -', term, ':', term**3) for term in initialSet]), stringify('\nSmallest cube:', min(initialSet)))
					
		counter += 1
		

# -----------------------------
# -----------------------------

@SetupAndTime(63,'Number of n-digit integers which are an nth power:')
def problem63(**kwargs):

	# For a given term n >= 10, the number of digits in n^(x+1) is always greater than n^x.
	# This indicates that any n >= 10 cannot be included in this equation, as 10^2 = 100, which is 3 digits.
	# So the question really is:

	# When does the point exist at which [1, 9]^n stops overlapping by 1 digit?
	
	numTerms = 0
	
	for num in xrange(10):
		power = 1
		while len(str(num**power)) == power:
			numTerms += 1			
			power += 1
			
	return numTerms

# -----------------------------
# -----------------------------

@SetupAndTime(64,'Number of square roots under %(numterms) that have an odd period:', numterms=10000)
def problem64(**kwargs):
		
	numFound = 0
	numterms = kwargs['numterms']
	for i in xrange(1, numterms+1):
		if len(sqrtExpand(i))%2 == 0:
			numFound += 1
			
	return numFound
	
# -----------------------------
# -----------------------------

@SetupAndTime(65,'Sum of the digits in the numerator of term %(maxTerm):', maxTerm=100)
def problem65(**kwargs):

	maxTerm = kwargs['maxTerm']
	
	# Just to make some calculations faster, assume first two terms are known so we can
	# just apply the pattern below rather than needing extra fancy logic.
	lastTerm = (3, 1)
	termBeforeLast = (2, 1)
	
	termsFound = 2
	
	while termsFound < maxTerm:
	
		nextTerm = (0, 0)
	
		if termsFound % 3 == 2:
			multiplier = termsFound - (termsFound/3)
			nextTerm = (multiplier*lastTerm[0] + termBeforeLast[0], multiplier*lastTerm[1] + termBeforeLast[1])

		else:
			nextTerm = (lastTerm[0] + termBeforeLast[0], lastTerm[1] + termBeforeLast[1])
						
		termBeforeLast = lastTerm
		lastTerm = nextTerm
			
		termsFound += 1
				
	return stringify(lastTerm[0], '->', sum([int(z) for z in str(lastTerm[0])]))


# -----------------------------
# -----------------------------

@SetupAndTime(66,'The minimal solution of x^2 - dy^2 = 1 with greatest x, d <= %(maxD) is:', maxD=1000)
def problem66(**kwargs):

	# x^2 - Dy^2, search for largest x.
	# Obviously x and y have to be related around D so that x*x = d*y*y+1
	
	maxD = kwargs['maxD']
	greatestX, greatestY, greatestD = 0, 0, 0
	
	for d in xrange(1, maxD+1):
	
		if perfectSquare(d): continue
		
		rationalized = rationalize(d)

		# Check period-1 
		num, denom = rationalized[-1]
		
		if num > greatestX:
			greatestX, greatestY, greatestD = num, denom, d
	
	return stringify('x=', greatestX, 'd=', greatestD, 'y=', greatestY)

# -----------------------------
# -----------------------------

@SetupAndTime(68,'The maximum digit string for a %(numsides)-gon ring is:', numsides=5,lengthfilter = lambda x: len(x) == 16) 
def problem68(**kwargs):

	def matchNTerms(sumsToTry, usedTerms, numTermsRemaining, numSides):
		
		matches = []
		
		# We probably can't search every sum.  If we reach an index that makes it
		# impossible to find numTermsRemaining, then we can skip it.
		numTermsToSearch = len(sumsToTry) - numTermsRemaining
		
		for pos in xrange(numTermsToSearch):
		
			element = sumsToTry[pos]
			tUsedTerms = dict(usedTerms)

			tryThisElement = True
	
			for x in element:
				if x not in usedTerms:  tUsedTerms[x] = 1
				elif usedTerms[x] == 1: tUsedTerms[x] = 2
				else:                   tryThisElement = False
			
			if tryThisElement:
			
				tUsedTerms = dict(usedTerms)
				for x in element:
					if x not in tUsedTerms: tUsedTerms[x] = 0
					tUsedTerms[x] += 1
			
				# Base case
				if numTermsRemaining == 0:
				
					# Ensure that half the terms are unique, and half the terms have a count of 2.
					unique = 0
					twoCount = 0
					for x in tUsedTerms:
						if tUsedTerms[x] == 1:   unique   += 1
						elif tUsedTerms[x] == 2: twoCount += 1
						
					# Only add the value if our dict of values contains a correct pairing of shared and unshared terms.
					if unique == twoCount and unique == numSides:
						matches.append([[element], tUsedTerms])
					
				else:
					results = matchNTerms(sumsToTry[pos+1:], tUsedTerms, numTermsRemaining-1, numSides)
					
					if results is not None:
						for result in results:
							matches.append([[element] + [x for x in result[0]], result[1]])
			
		if len(matches) == 0: return None
		return matches
		
	def getRingWith(termOrdering, currentTermString, terms, termRepetitions, positionOn, priorPosition3, filter):
	
		candidate = None
		
		# Find which of the remaining terms contains the last element of the last node.
		
		uniqueTerm, nonUniqueTerm = None, None
		numUniques = 0
		
		# Filter out anything bad that could introduce random "None"s into the equation.
		
		for term in terms:
			if priorPosition3 in term:
				candidate = term
				
				for element in candidate:
					if termRepetitions[element] == 1: 
						uniqueTerm = element
						numUniques += 1
						
					elif element != priorPosition3:
						nonUniqueTerm = element
				break
				
		if candidate is None or uniqueTerm is None or nonUniqueTerm is None or numUniques != 1: return None
		
		replacementStr = [existing if pos != positionOn else uniqueTerm for existing, pos in zip(currentTermString, termOrdering)]
		positionOn += 1
		
		if len(terms) > 1:
			replacementStr = [existing if pos != positionOn else nonUniqueTerm for existing, pos in zip(replacementStr, termOrdering)]
			positionOn += 1
		
			# Recurse through the list again without the element we found to be valid for this position and see whatever matches it.
			remainingTerms = [term for term in terms if term != candidate]
			result = getRingWith(termOrdering, replacementStr, remainingTerms, termRepetitions, positionOn, nonUniqueTerm, filter)
			if result is None: return None
			else:              return result
			
		else:
		
			remainingTermSum = sum(terms[0])
		
			for eachCombination in list_chunks(replacementStr, 3):
				if sum(eachCombination) != remainingTermSum: return None
					
			# Find the combination with the lowest external node.
			z = [x for x in replacementStr[::3]]
			z = min(z)
			
			while replacementStr[0] != z:
				replacementStr = rotate_right(replacementStr, 3)
				
			retval = ''.join([str(x) for x in replacementStr])
			
			if filter(retval): return int(retval)
			else:              return None
			
				
		
		
	def solveRing(termOrdering, termList, filter):

		remainingTerms = list(termList[0])
		termRepetitions = termList[1]
		
		# Add the initial term to the list, do a binary check on the two positions to see which one works.
		# Ensure stuff cuts off early if you can.
		
		replacementStr = [0]*len(termOrdering)
		permutations = []
		
		# Results may have the correct number of terms used once and twice, but this doesn't mean we're dealing
		# with valid data.  So discard any rings that aren't valid.
		
		numUniques = 0

		for element in remainingTerms[0]:
			if termRepetitions[element] == 1:
				replacementStr = [existing if pos != 1 else element for existing, pos in zip(replacementStr, termOrdering)]
				numUniques += 1
			else:
				permutations.append(element)
				
		if numUniques != 1: return None
				
		remainingTerms = remainingTerms[1:]	
		a, b = permutations
		
		test1 = [existing if pos != 2 else a for existing, pos in zip(replacementStr, termOrdering)]
		test1 = [existing if pos != 3 else b for existing, pos in zip(test1, termOrdering)]
		
		test2 = [existing if pos != 2 else b for existing, pos in zip(replacementStr, termOrdering)]
		test2 = [existing if pos != 3 else a for existing, pos in zip(test2, termOrdering)]
				
		maxAB = getRingWith(termOrdering, test1, remainingTerms, termRepetitions, 4, b, filter)
		maxBA = getRingWith(termOrdering, test2, remainingTerms, termRepetitions, 4, a, filter)
		
		if maxAB is None and maxBA is None:     return None
		if maxAB is None and maxBA is not None: return maxBA
		if maxBA is None and maxAB is not None: return maxAB
		return max(maxAB, maxBA)
		
	def tryToResolve(currentCircle, sumsToTry, numsides, filter):

		results = matchNTerms(sumsToTry, dict(), numsides-1, numsides)
		if results is None: return None
		
		largest = 0
		
		for result in results:			
			largestResult = solveRing(currentCircle, result, filter)
			
			if largestResult is not None and largestResult > largest: 
				largest = largestResult
		
		return largestResult

		
	numsides = kwargs['numsides']
	maxterm = 2*numsides
	
	# Filter out result by length for the final .  Which means the result actually won't be the greatest value...
	
	#filter = lambda x: True
	filter = kwargs['lengthfilter']
	

	# Unique term, term shared with prior, term shared with next.
	combination = [1, 2, 3]
	counter = 4
	for j in xrange(1, numsides):
	
		# First term.
		combination.append(counter)
		counter += 1
		
		# Second term is a repeat of an existing term.
		combination.append(combination[(j*3)-1])
		
		# Third term repeats only if this is the last term.
		if j == numsides-1:
			combination.append(combination[1])
		else:
			combination.append(counter)
			counter += 1
			
	sums = dict()
	
	for i in xrange(1, maxterm+1):
		for j in xrange(1, maxterm+1):
			for k in xrange(1, maxterm+1):
				if i==j or i==k or j==k: continue
				
				z = [i, j, k]
				z.sort()
				
				ijk = i+j+k
				
				if ijk not in sums: sums[ijk] = set()
				sums[ijk].add(tuple(z))
				
	# Prune any sums that don't have at least numsides options.
	remove = []
	
	for s in sums:
		if len(sums[s]) < numsides:
			remove.append(s)
			
	for s in remove:
		del sums[s]
				
	
	# Assume that we can rotate a valid magic ring that we find, so don't bother trying to
	# order it here.  Just see what actually works. This means, pick n permutations and try to match them up
	# until either no matches or found, or until all matches are found.
	
	largest = 0
	
	for sumValue in sums:
		largestForSum = tryToResolve(combination, list(sums[sumValue]), numsides, filter)
		
		if largestForSum is not None and largestForSum > largest:
			largest = largestForSum
			
	return largest
			
# -----------------------------
# -----------------------------

@SetupAndTime(69,'The number with the highest n/phi(n) up to %(upperbound) is:', upperbound=1000000)
def problem69(**kwargs):
	upperbound = kwargs['upperbound']
			
	total = 1
	for prime in Prime.ordered:
		z = total*prime
		if z > upperbound: break
		total = z
		
	return total

# -----------------------------
# -----------------------------

@SetupAndTime(70,'Using two prime terms, the minimum n/phi(n) which is a permutation of itself is:', upperbound=10**7)
def problem70(**kwargs):

	upperbound = kwargs['upperbound']
	
	maxterm = 0
	minNOverPhi = 9999999999999999
	
	gcd = Prime.gcd

	primes = [p for p in Prime.primerange(1, int(math.sqrt(upperbound))*2)]
	lenOrdered = len(primes)
	
	for p1 in xrange(lenOrdered-1):
		for p2 in xrange(p1, lenOrdered):
		
			prime1 = primes[p1]
			prime2 = primes[p2]
			
			combined = prime1*prime2
			if combined > upperbound: break
			
			phi = (prime1 - 1) * (prime2 - 1)
			nOverPhi = float(combined) / phi
			
			if nOverPhi < minNOverPhi and stringPermutation(str(phi), str(combined)):
				maxterm = combined
				minNOverPhi = nOverPhi
	
	if maxterm == 0:
		return 'No such terms exist in this range.'
		
	else:
		return stringify('n=', maxterm, 'n/phi(n)=', minNOverPhi)
					
					
# -----------------------------
# -----------------------------
@SetupAndTime(70,'Leftmost fracion to %(initial_frac) using denominators up to %(maxD):', maxD=1000000, initial_frac = (3, 7))
def problem71(**kwargs):
		
	# Fraction immediately before 3/7
	initial_frac = kwargs['initial_frac']
	
	# Maximum N to search up to.
	maxD = kwargs['maxD']
	
	largestND = max(initial_frac)
	
	maxMultiplier = 1
	while largestND*(maxMultiplier*10) < maxD: 
		maxMultiplier *= 10
		
	# Pad the initial term with 0's such that it's still smaller than the maximum term.
	frac = [x*maxMultiplier for x in initial_frac]
	
	print frac, maxMultiplier
		
	# Obviously the terms are equal at the moment.  We need to find the first term below this!
	frac[0] -= 1
	ratio = float(maxD) / frac[1]
	num = int(ratio*frac[0])*1.0
	
	denom = maxD
	
	# Precalculations so we don't need to recalc this term a bunch.
	leftTermFloat = float(initial_frac[0]) / initial_frac[1]
	
	# How low can the denominator go...before becoming larger than the max term we've specified?
	while (num/denom < leftTermFloat):
		denom -= 1
	denom += 1
		
	# In case we're not dealing with reduced terms, be sure to reduce!
	fgcd = Prime.gcd(num, denom)
	reducedFrac = (int(num)/fgcd, denom/fgcd)

	return strfrac(reducedFrac)

# -----------------------------
# -----------------------------


def problem():
		
	#maxterm = 1000000
	#runningtotal = 0
	#
	#for i in xrange(2, maxterm+1):
	#	if Prime.isPrime(i):
	#		runningtotal += i-1
	#	else:
	#		runningtotal += mul(Prime.primefactors(i))
	#		
	#print 'When d<=', maxterm, ', the set would be made of ', runningtotal, 'elements.'
	
	#phifunc = Prime.phi
	#for j in xrange(2, 2001):
	#	z = phifunc(j)
	#	if j%100 == 0:
	#		print j, z
	#		
	
	
	numprimes = 0
	for j in xrange(1,10**7):
		if Prime.isPrime(j):
			numprimes += 1
			
	print 'Num Primes:', numprimes

	

	# 2 - 1
	# 3 - 3
	# 4 - 5 - 2
	# 5 - 9
	# 6 - 11 - 2
	# 7 - 17
	# 8 - 21 - 4
	# 9 - 27 - 6
	# 10 - 31 - 4
	# 11 - 41
	# 12 - 45 - 4
	# 13 - 57
	# 14 - 63 - 7
	# 15 - 71 - 8
	# 16 - 79 - 8
	# 17 - 95
	# 18 - 101 - 6

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

# -----------------------------
# -----------------------------

def main(args):
	maxproblem = 70

	problems = range(1, maxproblem+1)

	totalTime = 0.0
	
	if len(args) > 0:
		problems = []
		for arg in args:
			problems.append(int(arg))
			
	for problem in problems:
		
		print 'Problem', problem

		result = eval('problem' + str(problem))()
		
		totalTime += result
		print ''
	
	print '\n//------------//-----------//---------//----------//\n'	
	print 'Total time spent doing problems:', totalTime, 'seconds.'

if __name__ == '__main__':

	start = time.clock()
	args = sys.argv[1:]
	
	if 'noprime' not in args: Prime.warmup(85000000)
	
	args = [a for a in args if not a == 'noprime']
	
	main(args)
	
	end = time.clock()
	
	print '\nTotal execution time:', end-start, 'seconds.'



		