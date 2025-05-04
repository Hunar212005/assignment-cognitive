# ---------------------------------------
# 1. LIST OPERATIONS
# ---------------------------------------
print("\n--- 1. LIST OPERATIONS ---")
L = [10, 20, 30, 40, 50, 60, 70, 80]

# i. Add 200 and 300
L.extend([200, 300])
print("After adding 200 & 300:", L)

# ii. Remove 10 and 30
L.remove(10)
L.remove(30)
print("After removing 10 & 30:", L)

# iii. Sort ascending
L.sort()
print("Ascending order:", L)

# iv. Sort descending
L.sort(reverse=True)
print("Descending order:", L)


# ---------------------------------------
# 2. TUPLE OPERATIONS
# ---------------------------------------
print("\n--- 2. TUPLE OPERATIONS ---")
scores = (45, 89.5, 76, 45.4, 89, 92, 58, 45)

# i. Highest score & index
max_score = max(scores)
print("Highest Score:", max_score, "| Index:", scores.index(max_score))

# ii. Lowest score & count
min_score = min(scores)
print("Lowest Score:", min_score, "| Count:", scores.count(min_score))

# iii. Reverse and convert to list
reversed_list = list(scores[::-1])
print("Reversed Tuple as List:", reversed_list)

# iv. Check if 76 is present
search_score = 76
if search_score in scores:
    print(f"{search_score} is present at index {scores.index(search_score)}")
else:
    print(f"{search_score} is not present in the tuple.")


# ---------------------------------------
# 3. RANDOM NUMBER ANALYSIS
# ---------------------------------------
print("\n--- 3. RANDOM NUMBER ANALYSIS ---")
import random

numbers = [random.randint(100, 900) for _ in range(100)]

# i. Odd numbers
odds = [n for n in numbers if n % 2 != 0]
print("Odd Numbers Count:", len(odds))

# ii. Even numbers
evens = [n for n in numbers if n % 2 == 0]
print("Even Numbers Count:", len(evens))

# iii. Prime numbers
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

primes = [n for n in numbers if is_prime(n)]
print("Prime Numbers Count:", len(primes))


# ---------------------------------------
# 4. SET OPERATIONS
# ---------------------------------------
print("\n--- 4. SET OPERATIONS ---")
A = {34, 56, 78, 90}
B = {78, 45, 90, 23}

# i. Union
print("Union:", A | B)

# ii. Intersection
print("Intersection:", A & B)

# iii. Symmetric Difference
print("Symmetric Difference:", A ^ B)

# iv. Subset & Superset
print("Is A subset of B?", A.issubset(B))
print("Is B superset of A?", B.issuperset(A))

# v. Remove element X from A
X = int(input("Enter a score to remove from set A: "))
if X in A:
    A.remove(X)
    print(f"{X} removed. Updated A:", A)
else:
    print(f"{X} not found in set A.")


# ---------------------------------------
# 5. DICTIONARY KEY RENAME
# ---------------------------------------
print("\n--- 5. DICTIONARY KEY RENAME ---")
location_dict = {"city": "New York", "country": "USA", "population": 8500000}

# Rename 'city' to 'location'
if "city" in location_dict:
    location_dict["location"] = location_dict.pop("city")

print("Updated Dictionary:", location_dict)
