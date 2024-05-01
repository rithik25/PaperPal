// script.js
async function submitQuery() {
    const userInput = document.getElementById('userInput').value;
    const responseContainer = document.getElementById('responseContainer');
    var response = document.getElementById("responseContainer");
    // esponse.textContent = userInput;

    try {
        const response = await fetch('http://localhost:5000/submit_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ userInput: userInput })
        });

        const result = await response.json();  // Parse JSON response
        responseContainer.textContent = result.response;  // Display the response from the server
    } catch (error) {
        responseContainer.textContent = 'Error: ' + error.message;
    }



    // try {
    //     const response = await fetch('API_URL', {
    //         method: 'POST',
    //         headers: {
    //             'Content-Type': 'application/json',
    //             'Authorization': 'Bearer YOUR_API_KEY'
    //         },
    //         body: JSON.stringify({
    //             prompt: userInput,
    //             max_tokens: 150
    //         })
    //     });

    //     if (!response.ok) {
    //         throw new Error('Failed to fetch response from the API');
    //     }

    //     const data = await response.json();
    //     responseContainer.textContent = data.choices[0].text;
    // } catch (error) {
    //     responseContainer.textContent = 'Error: ' + error.message;
    // }
}

