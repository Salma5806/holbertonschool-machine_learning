-- Lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
-- INNER JOIN WITH GROUP BY
SELECT g.genre AS genre, COUNT(s.id) AS number_of_shows
FROM genres g
JOIN shows s ON g.id = s.genre_id
GROUP BY g.genre
ORDER BY number_of_shows DESC;
