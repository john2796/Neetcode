# TOP SQL 50

https://leetcode.com/studyplan/top-sql-50/
https://www.w3schools.com/sql/sql_null_values.asp
https://www.mindmeister.com/app/map/3331253082

## 1. What is SQL ?

SQL (Structured Query Language) is a standardized langauge used to interact with relational databases. It allows users to query, manipulate, and manage data stored in relational database management system (RDBMS).

## 2. Key Components of SQL

**a. DDL (Data Definition Language)**
DDL is used to define and manage the structure of database objects

- CREATE TABLE: Creates a new Table

```sql
CREATE TABLE Students (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    age INT,
    grade CHAR(1)
)
```

- ALTER TABLE: Modifies an existing table (e.g., add/remove columns).

```sql
ALTER TABLE Students
ADD COLUMN city VARCHAR(50);
```

- DROP TABLE: Deletes an existing table.

```sql
DROP TABLE Students;
```

**b. DML (Data Manipulation Language)**
DML is used to manipulate data within tables.

- INSERT INTO: Inserts new rows into a table.

```sql
INSERT INTO Students (id, name, age, grade, city)
VALUES (1, 'John', 20, 'A', 'New York');
```

- UPDATE: Modifies existing records in a table.

```sql
UPDATE Students
SET age = 21
WHERE id = 1;
```

- DELETE: Removes existing records from a table.

```sql
DELETE FROM Students
WHERE id = 1;
```

**c. DQL (Data Query Language)**
DQL is used to retrieve data from a database.

- SELECT: Retrieves data from one or more tables.

```sql
SELECT * FROM Students;
```

- WHERE: Filters records based on specified conditions.

```sql
SELECT * FROM Students
WHERE age > 18;
```

- ORDER BY: Sorts the result set in ascending or descending order.

```sql
SELECT * FROM Students
ORDER BY age DESC;
```

- GROUP BY: Groups rows sharing a property to apply aggregate functions.

```sql
SELECT grade, COUNT(*)
FROM Students
GROUP BY grade;
```

- HAVING: Filters groups returned by a GROUP BY clause.

```sql
SELECT grade, COUNT(*)
FROM Students
GROUP BY grade
HAVING COUNT(*) > 1;
```

**d. DCL (Data Control Language)**
DCL is used to manage access permissions and control privileges.

- GRANT: Grants specific privileges to database users.

```sql
GRANT SELECT ON Students TO user1;
```

- REVOKE: Revokes previously granted permissions from database users.

```sql
REVOKE SELECT ON Students FROM user1;
```

## 3. Common Data Types in SQL

SQL supports various data types that define the type of data a column can hold. Common data types include:

- **Numeric Types**: `INT`, `FLOAT`, `DECIMAL`
- **Character Strings**: `VARCHAR`, `CHAR`
- **Date and Time Types**: `DATE`, `TIME`, `DATETIME`
- **Boolean**: `BOOLEAN`
- **Binary Data**: `BLOB`

## 4. SQL Constraints

Constraints enforce rules on data columns to ensure data integrity:

- **PRIMARY KEY**: Uniquely identifies each record in a table.
- **FOREIGN KEY**: Establishes a link between two tables.
- **NOT NULL**: Ensures a column cannot have NULL values.
- **UNIQUE**: Ensures all values in a column are different.
- **CHECK**: Ensures all values in a column satisfy a specific condition.

## 5. Relationship in SQL

- **On-to-One**: Each record in Table A is related to one record in table B
- **One-to-Many**: Each record in Table A can be related to multiple records in table B.
- **Many-to-Many**: Multiples records in Table A can be related to multiple records in Table B through an intermediary table.

## 6. SQL Joins

Joins combine rows from two or more tables based on related column:

- INNER JOIN: Returns records that have matching values in both tables.

```sql
SELECT Orders.OrderID, Customers.CustomerName
FROM Orders
INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID;
```

- LEFT JOIN (or LEFT OUTER JOIN): Returns all records from the left table and matching records from the right table.

```sql
SELECT Customers.CustomerName, Orders.OrderID
From Customers
LEFT JOIN Orders ON Customers.CustomerID = Orders.CustomerID;
```

- RIGHT JOIN (or RIGHT OUTER JOIN): Returns all records from the right table and matching records from the left table.

```sql
SELECT Customers.CustomerName, Orders.OrderID
From Customers
RIGHT JOIN Orders ON Customers.CustomerID = Orders.CustomerID;
```

- FULL JOIN (or FULL OUTER JOIN): Returns all records when there is a match in either left or right table.

```sql
SELECT Customers.CustomerName, Orders.OrderID
FROM Customers
FULL JOIN Orders ON Customers.CustomerID = Orders.CustomerID;
```

## 7. SQL Functions

SQL provides various built-in functions for manipulating data:

- Aggregate Functions: `COUNT()`, `SUM()`, `AVG()`, `MIN()`, `MAX()`

```sql
SELECT COUNT(*), AVG(Salary)
FROM Employees;
```
