# DOCUMENTED BEHAVIOR

## Stated Contract Purpose and Functionality

- The contract is designed to provide a Savings Vault for crvUSD, an ERC4626 token, allowing users to earn a "risk-free" interest rate on the crvUSD stablecoin.
- The contract addresses the issue of scrvUSD losing its ERC4626 capabilities when bridged cross-chain by creating secondary scrvUSD markets on all chains where scrvUSD can be redeemed.
- The contract uses an oracle to provide the rate at which the price of the asset is increasing, ensuring that the pool works as expected.
- The contract aims to compute the growth rate in a safe (non-manipulable) and precise (no losses due to approximation) way.

## Described Business Logic and Security Model

- The contract uses an oracle to fetch scrvUSD vault parameters from Ethereum and provide them on other chains.
- The contract introduces different versions (`v0`, `v1`, `v2`) with varying assumptions to compute the growth rate.
  - `v0`: Replicates scrvUSD and returns the real price at some timestamp.
  - `v1`: Assumes no further interaction with scrvUSD, rewards are distributed as is, and no new deposits/withdrawals alter the rate.
  - `v2`: Assumes rewards denominated in crvUSD are equal across subsequent periods.
- The contract uses a blockhash oracle with specific assumptions:
  - It can be updated frequently with a mainnet blockhash that is no older than 30 minutes.
  - It can rarely provide an incorrect blockhash, but not an incorrect block number.
  - A new update with a fresh block number will correct the parameters.

## Documented Component Interactions

- The contract interacts with the scrvUSD vault on Ethereum to fetch parameters.
- The contract interacts with secondary scrvUSD markets on other chains.
- The contract uses an oracle to provide the rate at which the price of the asset is increasing.
- The contract uses a blockhash oracle to ensure the correctness of the parameters.

## Specified User Flows and Permissions

- Users can earn a "risk-free" interest rate on the crvUSD stablecoin.
- Users can bridge scrvUSD cross-chain, but the token loses its ERC4626 capabilities and becomes a plain ERC20 token.
- Users can redeem scrvUSD on secondary markets.
- The contract is controlled by a DAO, and its parameters can be changed by a vote.

## Defined System Constraints

- The contract assumes that the price of scrvUSD is always growing.
- The contract assumes that the rate is more or less stable over a 1-week period and small declines do not matter.
- The contract assumes that rewards denominated in crvUSD are equal across subsequent periods.
- The contract assumes that the oracle provides the correct rate at which the price of the asset is increasing.
- The contract assumes that the blockhash oracle provides a correct blockhash.

# SECURITY SPECIFICATIONS

## Documented Security Features

- The contract uses an oracle to provide the rate at which the price of the asset is increasing.
- The contract uses a blockhash oracle to ensure the correctness of the parameters.
- The contract uses smoothing to introduce sudden updates, so the price slowly catches up with the price, while the pool is being arbitraged safely.

## Stated Access Control Rules

- The contract is controlled by a DAO, and its parameters can be changed by a vote.

## Described Risk Mitigations

- The contract uses a blockhash oracle to ensure the correctness of the parameters.
- The contract uses smoothing to introduce sudden updates, so the price slowly catches up with the price, while the pool is being arbitraged safely.
- The contract assumes that the oracle provides the correct rate at which the price of the asset is increasing.
- The contract assumes that the blockhash oracle provides a correct blockhash.

## Emergency Procedures

- No emergency procedures are documented.

## Known Limitations

- The contract assumes that the price of scrvUSD is always growing.
- The contract assumes that the rate is more or less stable over a 1-week period and small declines do not matter.
- The contract assumes that rewards denominated in crvUSD are equal across subsequent periods.
- The contract assumes that the oracle provides the correct rate at which the price of the asset is increasing.
- The contract assumes that the blockhash oracle provides a correct blockhash.

# SYSTEM ARCHITECTURE

## Documented System Components

- Savings Vault for crvUSD
- Secondary scrvUSD markets
- Oracle to provide the rate at which the price of the asset is increasing
- Blockhash oracle to ensure the correctness of the parameters

## Described Integration Points

- The contract interacts with the scrvUSD vault on Ethereum to fetch parameters.
- The contract interacts with secondary scrvUSD markets on other chains.
- The contract uses an oracle to provide the rate at which the price of the asset is increasing.
- The contract uses a blockhash oracle to ensure the correctness of the parameters.

## Stated Economic Model

- The contract allows users to earn a "risk-free" interest rate on the crvUSD stablecoin.
- The contract uses a blockhash oracle to ensure the correctness of the parameters.
- The contract uses smoothing to introduce sudden updates, so the price slowly catches up with the price, while the pool is being arbitraged safely.

## Listed Deployment Parameters

- No deployment parameters are listed.

## Configuration Requirements

- The contract is controlled by a DAO, and its parameters can be changed by a vote.

# IMPLICIT ELEMENTS

## Unstated Assumptions

- The contract assumes that the oracle provides the correct rate at which the price of the asset is increasing.
- The contract assumes that the blockhash oracle provides a correct blockhash.
- The contract assumes that the price of scrvUSD is always growing.
- The contract assumes that the rate is more or less stable over a 1-week period and small declines do not matter.
- The contract assumes that rewards denominated in crvUSD are equal across subsequent periods.

## Unclear Specifications

- The documentation does not specify how the oracle provides the rate at which the price of the asset is increasing.
- The documentation does not specify how the blockhash oracle ensures the correctness of the parameters.
- The documentation does not specify how the contract handles sudden updates to the price.

## Ambiguous Behaviors

- The documentation does not specify how the contract handles sudden updates to the price.
- The documentation does not specify how the contract handles incorrect blockhashes provided by the blockhash oracle.

## Documentation Gaps

- The documentation does not specify how the oracle provides the rate at which the price of the asset is increasing.
- The documentation does not specify how the blockhash oracle ensures the correctness of the parameters.
- The documentation does not specify how the contract handles sudden updates to the price.
- The documentation does not specify how the contract handles incorrect blockhashes provided by the blockhash oracle.
- The documentation does not specify any emergency procedures.
- The documentation does not list any deployment parameters.