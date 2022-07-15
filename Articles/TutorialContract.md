---
title: Smart Contract Tutorial 001
tags:["tutorial"]
excerpt:Four steps to deploy your blockchain with contract
date: 2022-07-15
---


## Four steps to deploy your blockchain with contract

### Get your local Ethereum networking running
- Install local server **Hardhat**
```echo
mkdir my-wave-portal
cd my-wave-portal
npm init -y
npm install --save-dev hardhat
```
- Get basic project
```echo
npx hardhat
//install hardhat-waffle and hardhat-ethers
npm install --save-dev @nomiclabs/hardhat-waffle ethereum-waffle chai @nomiclabs/ 
//print out a bunch of strings ,Ethereum address
npx hardhat accounts 
//run it
npx hardhat compile
npx hardhat test
```
- Delete sample-test.js under test, sample-script.js under scripts, Greeter.sol under contracts 

### Write first smart contract in Solidity
- Create a file named WavePortal.sol under contracts directory
```solidity
// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.0;

import "hardhat/console.sol";

contract WavePortal {
    constructor() {
        console.log("Yo yo, I am a contract and I am smart");
    }
}
```

### Compile contract locally and run it
- Go into the scripts directory and make a file named run.js
```javascript
const main = async () => {
	const waveContractFactory = await hre.ethers.getContractFactory("wavePortal");
	// create a local Ethereum network just for this contract
	const waveContract = await waveContractFactory.deploy();
	await waveContract.deployed();
	console.log("Contract deployed to:", waveContract.address);
};

const runMain = async () => {
	try{
	await main();
	// exit Node process without error
	process.exit(0);
	} catch (error) {
	// exit Node process while indicating 'Uncaught Fatal Exception' error
	console.log(error);
	process.exit(1);
	}
};
runMain();
```
-  Run it
```
npx hardhat run scripts/run.js
```
### Store data on smart contract
- Add to functions to "store" waves
```solidity
uint256 totalWaves;
function wave() public {
	totalWaves += 1;
	console.log("%s has waved", meg.sender);
}
function getTotalWaves() public view returns (uint256) {
	console.log("we have %d total waves", totalWaves);
	return totalWaves;
}
```
- Now change the main part
```javascript
const main = async () => {
  //
  const [owner, randomPerson] = await hre.ethers.getSigners();
  // which is the same as before
  const waveContractFactory = await hre.ethers.getContractFactory("WavePortal");
  const waveContract = await waveContractFactory.deploy();
  await waveContract.deployed();

  console.log("Contract deployed to:", waveContract.address);
  console.log("Contract deployed by:", owner.address);
  // call the function to grab the # of total waves
  let waveCount;
  waveCount = await waveContract.getTotalWaves();
  //do the wave
  let waveTxn = await waveContract.wave();
  await waveTxn.wait();
  //grab the waveCount again
  waveCount = await waveContract.getTotalWaves();
};
```
-  Run it
```
npx hardhat run scripts/run.js
```
- Add to "main" part
```javascript
  //simulate other people hitting
  waveTxn = await waveContract.connect(randomPerson).wave();
  await waveTxn.wait();

  waveCount = await waveContract.getTotalWaves();
```
### Deploy locally
- keep the network alive 
```
npx hardhat node
```
- create deploy.js under scripts
```javascript
const main = async () => {
  const [deployer] = await hre.ethers.getSigners();
  const accountBalance = await deployer.getBalance();

  console.log("Deploying contracts with account: ", deployer.address);
  console.log("Account balance: ", accountBalance.toString());
  //same as run.js compile, deploy, execute
  console.log("WavePortal address: ", waveContract.address);
};
const runMain = async () => {}
runMain();
```
- From a different terminal window deploy
```
npx hardhat run scripts/deploy.js --network localhost
```
### Setup a basic react app, setup Metamask
- Use replit
- Connect Wallet 
### Deploy smart contract to a real testnet
- Testnets
	+ Broadcast our transaction
	+ Wait for it to be picked up by actual miners
	+ Wait for it to be mined
	+ Wait for it to be broadcasted back to the blockchain telling all the other miners to update their copies
- Getting some fake from
	+ [Chainlink](https://faucets.chain.link/rinkeby)
	+ [Official Rinkeby](https://faucet.rinkeby.io/)
- Deploy to Rinkeby testnet
```javascript
require("@nomiclabs/hardhat-waffle");

module.exports = {
  solidity: "0.8.0",
  networks: {
    rinkeby: {
      url: "YOUR_ALCHEMY_API_URL",
      accounts: ["YOUR_PRIVATE_RINKEBY_ACCOUNT_KEY"]
    },
  },
};
```
- Run this command from the root 
```
npx hardhat run scripts/deploy.js --network rinkeby
// the result is like belows
Deploying contracts with the account: 0xF79A3bb8d5b93686c4068E2A97eAeC5fE4843E7D
Account balance: 3198297774605223721
WavePortal address: 0xd5f08a0ae197482FA808cE84E00E97d940dBD26E
```
### Connect our wallet to web app
- Explain  React code with annotation
```javascript
import { ethers } from "ethers";
import abi from "./utils/WavePortal.json";
import React, { useEffect, useState } from "react";

import "./App.css";

const App = () => {
    const contractABI = abi.abi;
  /*
   * All state property to store all waves
   */
    const [allWaves, setAllWaves] = useState([]);
  /*
  * Just a state variable we use to store our user's public wallet.
  */
    const [currentAccount, setCurrentAccount] = useState("");
  /**
   * Create a variable here that holds the contract address after you deploy!
   */    
    const contractAddress = "0x77111112222233333111114bb2fDE1c8e0311111";

    const checkIfWalletIsConnected = async () => {
        try {
            const { ethereum } = window;

            if (!ethereum) {
                console.log("Make sure you have metamask!");
                return;
            } else {
                console.log("We have the ethereum object", ethereum);
            }
            /*
      		* Check if we're authorized to access the user's wallet
      		*/
            const accounts = await ethereum.request({ method: "eth_accounts" });

            if (accounts.length !== 0) {
                const account = accounts[0];
                console.log("Found an authorized account:", account);
                setCurrentAccount(account);
            } else {
                console.log("No authorized account found");
            }
        } catch (error) {
            console.log(error);
        }
    };

    /**
     * Implement connectWallet method here
     */
    const connectWallet = async () => {
        try {
            const { ethereum } = window;

            if (!ethereum) {
                alert("Get MetaMask!");
                return;
            }

            const accounts = await ethereum.request({ method: "eth_requestAccounts" });

            console.log("Connected", accounts[0]);
            setCurrentAccount(accounts[0]);
        } catch (error) {
            console.log(error);
        }
    };

	// officially reading data from your contract on the blockchain through your web client.
    const wave = async (message) => {
        try {
            const { ethereum } = window;

            if (ethereum) {
                const provider = new ethers.providers.Web3Provider(ethereum);
                const signer = provider.getSigner();
                /*
                * You're using contractABI here
                */
                const wavePortalContract = new ethers.Contract(contractAddress, contractABI, signer);
                let count = await wavePortalContract.getTotalWaves();
                console.log("Retrieved total wave count...", count.toNumber());
                /*
                 * Execute the actual wave from your smart contract
                 */
                const waveTxn = await wavePortalContract.wave(message, { gasLimit: 300000 });
                console.log("Mining...", waveTxn.hash);
                await waveTxn.wait();
                console.log("Mined -- ", waveTxn.hash);
                count = await wavePortalContract.getTotalWaves();
                console.log("Retrieved total wave count...", count.toNumber());
            } else {
                console.log("Ethereum object doesn't exist!");
            }
        } catch (error) {
            console.log(error);
        }
    };

    const getAllWaves = async () => {
        const { ethereum } = window;

        try {
            if (ethereum) {
                const provider = new ethers.providers.Web3Provider(ethereum);
                const signer = provider.getSigner();
                const wavePortalContract = new ethers.Contract(contractAddress, contractABI, signer);
                // call the getAllWaves method from your Smart Contract
                const waves = await wavePortalContract.getAllWaves();
				// pick out address, timestamp, message.
                const wavesCleaned = waves.map((wave) => {
                    return {
                        address: wave.waver,
                        timestamp: new Date(wave.timestamp * 1000),
                        message: wave.message,
                    };
                });
        /*
         * Store our data in React State
         */
                setAllWaves(wavesCleaned);
            } else {
                console.log("Ethereum object doesn't exist!");
            }
        } catch (error) {
            console.log(error);
        }
    };
/**
 * Listen in for emitter events!
 */
    useEffect(() => {
        checkIfWalletIsConnected();
    }, []);

    useEffect(() => {
        let wavePortalContract;

        const onNewWave = (from, timestamp, message) => {
            console.log("NewWave", from, timestamp, message);
            setAllWaves((prevState) => [
                ...prevState,
                {
                    address: from,
                    timestamp: new Date(timestamp * 1000),
                    message: message,
                },
            ]);
        };

        if (window.ethereum) {
            const provider = new ethers.providers.Web3Provider(window.ethereum);
            const signer = provider.getSigner();

            wavePortalContract = new ethers.Contract(contractAddress, contractABI, signer);
            wavePortalContract.on("NewWave", onNewWave);
        }

        return () => {
            if (wavePortalContract) {
                wavePortalContract.off("NewWave", onNewWave);
            }
        };
    }, []);

    return (
        <div className="mainContainer">
            <div className="dataContainer">
                <div className="header">👋 Hey there!</div>

                <div className="bio">
                    I am farza and I worked on self-driving cars so that's pretty cool right? Connect your Ethereum wallet and wave at me!
                </div>

                <button className="waveButton" onClick={() => wave("Hello World")}>
                    Wave at Me
                </button>

                {!currentAccount && (
                    <button className="waveButton" onClick={connectWallet}>
                        Connect Wallet
                    </button>
                )}

                {allWaves.map((wave, index) => {
                    return (
                        <div key={index} style={{ backgroundColor: "OldLace", marginTop: "16px", padding: "8px" }}>
                            <div>Address: {wave.address}</div>
                            <div>Time: {wave.timestamp.toString()}</div>
                            <div>Message: {wave.message}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default App;
```
### Storing messages from users on the blockchain
- Let users submit a message along with their wave.
- Have that data saved somehow on the blockchain.
- Show that data on our site so anyone can come to see all the people who have waved at us and their messages.
```solidity
// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.0;

import "hardhat/console.sol";

contract WavePortal {
    uint256 totalWaves;

    /*
     * A little magic, Google what events are in Solidity!
     */
    event NewWave(address indexed from, uint256 timestamp, string message);

    /*
     * I created a struct here named Wave.
     * A struct is basically a custom datatype where we can customize what we want to hold inside it.
     */
    struct Wave {
        address waver; // The address of the user who waved.
        string message; // The message the user sent.
        uint256 timestamp; // The timestamp when the user waved.
    }

    /*
     * I declare a variable waves that lets me store an array of structs.
     * This is what lets me hold all the waves anyone ever sends to me!
     */
    Wave[] waves;

    constructor() {
        console.log("I AM SMART CONTRACT. POG.");
    }

    /*
     * You'll notice I changed the wave function a little here as well and
     * now it requires a string called _message. This is the message our user
     * sends us from the frontend!
     */
    function wave(string memory _message) public {
        totalWaves += 1;
        console.log("%s waved w/ message %s", msg.sender, _message);

        /*
         * This is where I actually store the wave data in the array.
         */
        waves.push(Wave(msg.sender, _message, block.timestamp));

        /*
         * I added some fanciness here, Google it and try to figure out what it is!
         * Let me know what you learn in #general-chill-chat
         */
        emit NewWave(msg.sender, block.timestamp, _message);
        
    	uint256 prizeAmount = 0.0001 ether;
    	require(
        	prizeAmount <= address(this).balance,
        	"Trying to withdraw more money than the contract has."
    	);
    	(bool success, ) = (msg.sender).call{value: prizeAmount}("");
    	require(success, "Failed to withdraw money from contract.");        
        
    }

    /*
     * I added a function getAllWaves which will return the struct array, waves, to us.
     * This will make it easy to retrieve the waves from our website!
     */
    function getAllWaves() public view returns (Wave[] memory) {
        return waves;
    }

    function getTotalWaves() public view returns (uint256) {
        // Optional: Add this line if you want to see the contract print the value!
        // We'll also print it over in run.js as well.
        console.log("We have %d total waves!", totalWaves);
        return totalWaves;
    }
}
```
- Update to run.js
```javascript
const main = async () => {
  //same as run.js compile, deploy, execute,3 lines.
    const waveContract = await waveContractFactory.deploy({
    value: hre.ethers.utils.parseEther("0.1"),
  });
  //console.log("WavePortal address: ", waveContract.address);
    console.log("Contract addy:", waveContract.address);
    
      /*
   * Get Contract balance
   */
  let contractBalance = await hre.ethers.provider.getBalance(
    waveContract.address
  );
  console.log(
    "Contract balance:",
    hre.ethers.utils.formatEther(contractBalance)
  );
  
  // let waveCount;
  // waveCount = await waveContract.getTotalWaves();
  // console.log(waveCount.toNumber());

  /**
   * Let's send a few waves!
   */
  let waveTxn = await waveContract.wave("A message!");
  await waveTxn.wait(); // Wait for the transaction to be mined

  //const [_, randomPerson] = await hre.ethers.getSigners();
  //waveTxn = await waveContract.connect(randomPerson).wave("Another message!");
  //await waveTxn.wait(); // Wait for the transaction to be mined
  
  /*
   * Get Contract balance to see what happened!
   */
  contractBalance = await hre.ethers.provider.getBalance(waveContract.address);
  console.log(
    "Contract balance:",
    hre.ethers.utils.formatEther(contractBalance)
  );
  
  let allWaves = await waveContract.getAllWaves();
  console.log(allWaves);  
}
```
```
npx hardhat run scripts/run.js
```
- Re-deploy
	+ We need to deploy it again.
	+ We need to update the contract address on our frontend. Change contractAddress in App.js to be the new contract address we got from the step above in the terminal .
	+ We need to update the abi file on our frontend. 
- Update to deploy.js
```javascript
const main = async () => {
  const waveContractFactory = await hre.ethers.getContractFactory("WavePortal");
  const waveContract = await waveContractFactory.deploy({
    value: hre.ethers.utils.parseEther("0.001"),
  });

  await waveContract.deployed();

  console.log("WavePortal address: ", waveContract.address);
};
const runMain = async () => {};
runMain();
```
```
npx hardhat run scripts/deploy.js --network rinkeby
```
### Randomly pick a winner and prevent spammers
```solidity
    /*
     * We will be using this below to help generate a random number
     */
    uint256 private seed;
    constructor() payable {
        console.log("We have been constructed!");
        /*
         * Set the initial seed
         */
        seed = (block.timestamp + block.difficulty) % 100;
    }
    function wave(string memory_message) public {
    
            /*
         * We need to make sure the current timestamp is at least 15-minutes bigger than the last timestamp we stored
         */
        require(
            lastWavedAt[msg.sender] + 15 minutes < block.timestamp,
            "Wait 15m"
        );
        
                /*
         * Update the current timestamp we have for the user
         */
        lastWavedAt[msg.sender] = block.timestamp;
            /*
         * Generate a new seed for the next user that sends a wave
         */
        seed = (block.difficulty + block.timestamp + seed) % 100;
        
                if (seed <= 50) {
            console.log("%s won!", msg.sender);

            uint256 prizeAmount = 0.0001 ether;
            require(
                prizeAmount <= address(this).balance,
                "Trying to withdraw more money than they contract has."
            );
            (bool success, ) = (msg.sender).call{value: prizeAmount}("");
            require(success, "Failed to withdraw money from contract.");
        }

        console.log("Random # generated: %d", seed);
    }
```