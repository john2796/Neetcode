# TOP SQL 50

 https://leetcode.com/studyplan/top-sql-50/
 https://www.w3schools.com/sql/sql_null_values.asp

1. What is SQL ?
SQL (Structured Query Language) is a standardized langauge used to interact with relational databases. It allows users to query, manipulate, and manage data stored in relational database management system (RDBMS).

2. Key Components of SQL

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
